"""对抗样本检测：score 越大表示越异常，calibrate 使用独立干净数据设阈值。"""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from .features import activations, pooled
from .preprocessing import preprocess


def device_of(model):
    return next(model.parameters()).device


class Detector:
    def __init__(self, model, **options):
        self.model = model.eval()
        self.options = options
        self.threshold = None

    def fit(self, loader, **options):
        return self

    def calibrate(self, loader, false_positive_rate=0.05):
        scores = []
        for images, _ in loader:
            with torch.no_grad():
                scores.append(self.score(images.to(device_of(self.model))).cpu())
        values = torch.cat(scores)
        self.threshold = float(torch.quantile(values, 1 - false_positive_rate, interpolation="higher"))
        self.calibration = dict(samples=len(values), target_fpr=false_positive_rate,
                                observed_fpr=float((values > self.threshold).float().mean()))
        return self

    def predict(self, images):
        if self.threshold is None:
            raise ValueError("请先使用独立干净数据调用 calibrate。")
        return self.score(images) > self.threshold

    def save(self, path):
        state = {key: value for key, value in self.__dict__.items() if key not in ("model", "regressor")}
        if hasattr(self, "regressor"):
            state["regressor_state"] = self.regressor.state_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(method=self.__class__.__name__, state=state), path)


class Confidence(Detector):
    def score(self, images):
        return 1 - self.model(images).softmax(1).max(1).values


class Entropy(Detector):
    def score(self, images):
        probabilities = self.model(images).softmax(1)
        return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1) / math.log(probabilities.shape[1])


class FeatureSqueezing(Detector):
    """位深压缩与中值平滑的联合检测，取概率向量 L1 距离的最大值。"""
    def __init__(self, model, bits=5, kernel=3):
        super().__init__(model, bits=bits, kernel=kernel)

    def score(self, images):
        probabilities = self.model(images).softmax(1)
        scores = []
        for transform in [dict(kind="bit_depth", bits=self.options["bits"]), dict(kind="median", kernel=self.options["kernel"])]:
            inputs = preprocess(images, [transform], bpda=torch.is_grad_enabled())
            scores.append((probabilities - self.model(inputs).softmax(1)).abs().sum(1))
        return torch.stack(scores).max(0).values


class Consistency(Detector):
    def score(self, images):
        inputs = preprocess(images, [dict(kind="median")], bpda=torch.is_grad_enabled())
        return 0.5 * (self.model(images).softmax(1) - self.model(inputs).softmax(1)).abs().sum(1)


class Mahalanobis(Detector):
    """类条件高斯特征距离；可用带标签的独立检测集拟合多层融合器。"""
    def __init__(self, model, layers, regularization=1e-4, magnitude=0.0):
        super().__init__(model, layers=list(layers), regularization=regularization, magnitude=magnitude)

    def fit(self, loader, **options):
        # 流式累计每类特征和与二阶矩，不保存整套训练特征。
        sums, moments, counts = None, None, None
        device = device_of(self.model)
        for images, labels in loader:
            with torch.no_grad():
                logits, features = activations(self.model, images.to(device), self.options["layers"])
                features = [pooled(value).double() for value in features]
            labels = labels.to(device)
            if sums is None:
                classes = logits.shape[1]
                counts = torch.zeros(classes, device=device, dtype=torch.float64)
                sums = [value.new_zeros(classes, value.shape[1]) for value in features]
                moments = [value.new_zeros(value.shape[1], value.shape[1]) for value in features]
            counts += torch.bincount(labels, minlength=len(counts))
            for index, value in enumerate(features):
                sums[index].index_add_(0, labels, value)
                moments[index] += value.T @ value
        if bool((counts == 0).any()):
            raise ValueError("拟合数据需要覆盖每个类别。")
        self.means, self.precisions = [], []
        for total, second in zip(sums, moments):
            mean = total / counts[:, None]
            covariance = (second - mean.T @ (mean * counts[:, None])) / counts.sum()
            covariance += torch.eye(covariance.shape[0], device=device) * self.options["regularization"]
            self.means.append(mean.float())
            self.precisions.append(torch.linalg.pinv(covariance, hermitian=True).float())
        self.fusion_weight = None
        return self

    def layer_scores(self, images):
        _, features = activations(self.model, images, self.options["layers"])
        scores = []
        for value, means, precision in zip(features, self.means, self.precisions):
            value = pooled(value)
            distances = []
            for mean in means:
                residual = value - mean
                distances.append((residual @ precision * residual).sum(1))
            scores.append(0.5 * torch.stack(distances, 1).min(1).values)
        return torch.stack(scores, 1)

    def fit_fusion(self, clean_images, adversarial_images):
        """在检测器开发集拟合逻辑回归；输入张量需与测试集分开。"""
        from sklearn.linear_model import LogisticRegression
        with torch.no_grad():
            scores = torch.cat([self.layer_scores(clean_images), self.layer_scores(adversarial_images)]).cpu().numpy()
        labels = [0] * len(clean_images) + [1] * len(adversarial_images)
        fusion = LogisticRegression(max_iter=1000).fit(scores, labels)
        self.fusion_weight = torch.tensor(fusion.coef_[0], device=device_of(self.model), dtype=torch.float32)
        self.fusion_bias = float(fusion.intercept_[0])
        return self

    def score(self, images):
        if self.options["magnitude"]:
            with torch.enable_grad():
                probe = images.detach().requires_grad_(True)
                gradient = torch.autograd.grad(self.layer_scores(probe).sum(), probe)[0]
            images = (images - self.options["magnitude"] * gradient.sign()).clamp(0, 1)
        scores = self.layer_scores(images)
        if self.fusion_weight is not None:
            return (scores @ self.fusion_weight + self.fusion_bias).sigmoid()
        return scores.mean(1)


class LayerRegression(Detector):
    """从指定中间层的中间 60% 特征回归深层特征，以 MSE 为检测分数。"""
    def __init__(self, model, layers, target_layer, hidden_dim=2048):
        super().__init__(model, layers=list(layers), target_layer=target_layer, hidden_dim=hidden_dim)

    def vectors(self, images):
        logits, values = activations(self.model, images, self.options["layers"] + [self.options["target_layer"]])
        parts = []
        for value in values[:-1]:
            flat = value.flatten(1)
            parts.append(flat[:, int(flat.shape[1] * 0.2):int(flat.shape[1] * 0.8)])
        return logits, torch.cat(parts, 1), pooled(values[-1])

    def build_regressor(self):
        hidden = self.options["hidden_dim"]
        self.regressor = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, self.target_dim)).to(device_of(self.model))

    def fit(self, loader, epochs=10, learning_rate=0.0003, **options):
        device = device_of(self.model)
        with torch.no_grad():
            _, vector, target = self.vectors(next(iter(loader))[0].to(device))
        self.input_dim, self.target_dim = vector.shape[1], target.shape[1]
        self.build_regressor()
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=learning_rate)
        self.history = []
        for epoch in range(epochs):
            total, count = 0.0, 0
            for images, labels in loader:
                with torch.no_grad():
                    logits, vector, target = self.vectors(images.to(device))
                    correct = logits.argmax(1).eq(labels.to(device))
                    vector, target = vector[correct], target[correct]
                if not len(vector):
                    continue
                optimizer.zero_grad()
                loss = F.mse_loss(self.regressor(vector), target)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(vector)
                count += len(vector)
            if not count:
                raise ValueError("拟合集中没有分类正确的样本，请检查模型权重与预处理。")
            self.history.append(dict(epoch=epoch + 1, loss=total / count, samples=count))
        self.regressor.eval()
        return self

    def score(self, images):
        _, vector, target = self.vectors(images)
        return (self.regressor(vector) - target).square().mean(1)


def load_detector(path, model):
    from .anomaly import Energy, InputStatistics
    artifact = torch.load(path, map_location=device_of(model), weights_only=True)
    constructors = {cls.__name__: cls for cls in (Confidence, Entropy, Consistency, FeatureSqueezing, Mahalanobis, LayerRegression, Energy, InputStatistics)}
    state = artifact["state"]
    detector = constructors[artifact["method"]](model, **state["options"])
    regressor_state = state.pop("regressor_state", None)
    detector.__dict__.update(state)
    if regressor_state is not None:
        detector.build_regressor()
        detector.regressor.load_state_dict(regressor_state)
        detector.regressor.eval()
    return detector
