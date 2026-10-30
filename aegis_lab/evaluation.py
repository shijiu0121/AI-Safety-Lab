"""分类、检测和攻防评估，结果为可序列化字典。"""
import time
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from .attacks import attack
from .detection import device_of


def classification_metrics(labels, predictions, adversarial_predictions, num_classes):
    labels = np.asarray(labels)
    clean = np.asarray(predictions) == labels
    attacked_correct = np.asarray(adversarial_predictions) == labels
    robust = clean & attacked_correct
    per_class = []
    for category in range(num_classes):
        selected = labels == category
        if selected.any():
            per_class.append(dict(class_id=category, samples=int(selected.sum()), clean_accuracy=float(clean[selected].mean()), robust_accuracy=float(robust[selected].mean())))
    return dict(samples=len(labels), clean_accuracy=float(clean.mean()), robust_accuracy=float(robust.mean()), attacked_accuracy=float(attacked_correct.mean()),
                balanced_clean_accuracy=float(np.mean([row["clean_accuracy"] for row in per_class])),
                balanced_robust_accuracy=float(np.mean([row["robust_accuracy"] for row in per_class])),
                attack_success_rate=float((clean & ~robust).sum() / clean.sum()) if clean.any() else None,
                per_class=per_class)


def detection_metrics(clean_scores, abnormal_scores, threshold):
    clean, abnormal = np.asarray(clean_scores), np.asarray(abnormal_scores)
    labels = np.r_[np.zeros(len(clean)), np.ones(len(abnormal))]
    values = np.r_[clean, abnormal]
    fpr, tpr, _ = roc_curve(labels, values)
    return dict(auroc=float(roc_auc_score(labels, values)), fpr=float((clean > threshold).mean()),
                tpr=float((abnormal > threshold).mean()), threshold=float(threshold),
                fpr_at_95_tpr=float(fpr[np.flatnonzero(tpr >= 0.95)[0]]),
                clean_samples=len(clean), abnormal_samples=len(abnormal),
                roc=[dict(fpr=float(x), tpr=float(y)) for x, y in zip(fpr, tpr)])


def evaluate(model, loader, attacks, *, case_callback=None):
    model.eval()
    device = device_of(model)
    results = []
    for definition in attacks:
        options = dict(definition)
        method = options.pop("method")
        labels, predictions, adversarial_predictions = [], [], []
        distances, elapsed, cases = [], 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            adversarial = attack(model, images, targets, method, **options)
            with torch.no_grad():
                clean_logits, adversarial_logits = model(images), model(adversarial)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            clean_pred, adv_pred = clean_logits.argmax(1), adversarial_logits.argmax(1)
            labels.extend(targets.tolist())
            predictions.extend(clean_pred.tolist())
            adversarial_predictions.extend(adv_pred.tolist())
            distances.extend((adversarial - images).flatten(1).abs().max(1).values.tolist())
            if case_callback and cases < 4:
                selected = torch.where(clean_pred.eq(targets) & adv_pred.ne(targets))[0][:4 - cases]
                for index in selected.tolist():
                    case_callback(definition, images[index].cpu(), adversarial[index].cpu(), int(targets[index]), int(clean_pred[index]), int(adv_pred[index]))
                    cases += 1
        metrics = classification_metrics(labels, predictions, adversarial_predictions, clean_logits.shape[1])
        results.append(dict(attack=method, parameters=definition, **metrics, mean_linf=float(np.mean(distances)), seconds=elapsed, milliseconds_per_sample=1000 * elapsed / len(labels)))
    return results


def adaptive_attack(model, detector, images, labels, *, epsilon_255=8, steps=100, step_size_255=2, weight=1.0, seed=0):
    """PGD 联合最大化分类损失并降低检测分数；逐样本保留成功且未告警的结果。"""
    generator = torch.Generator(device=images.device).manual_seed(seed)
    epsilon = epsilon_255 / 255
    current = (images + (torch.rand(images.shape, generator=generator, device=images.device) * 2 - 1) * epsilon).clamp(0, 1)
    best = current.detach().clone()
    best_objective = images.new_full((len(images),), -torch.inf)
    best_success = torch.zeros(len(images), dtype=torch.bool, device=images.device)
    for _ in range(steps):
        current = current.detach().requires_grad_(True)
        logits = model(current)
        objective = torch.nn.functional.cross_entropy(logits, labels, reduction="none") - weight * torch.relu(detector.score(current) - detector.threshold)
        gradient = torch.autograd.grad(objective.sum(), current)[0]
        current = current.detach() + step_size_255 / 255 * gradient.sign()
        current = torch.maximum(torch.minimum(current, images + epsilon), images - epsilon).clamp(0, 1)
        with torch.no_grad():
            logits = model(current)
            scores = detector.score(current)
            objective = torch.nn.functional.cross_entropy(logits, labels, reduction="none") - weight * torch.relu(scores - detector.threshold)
            success = logits.argmax(1).ne(labels) & (scores <= detector.threshold)
            replace = (success & ~best_success) | ((success == best_success) & (objective > best_objective))
            best = torch.where(replace[:, None, None, None], current, best)
            best_objective = torch.where(replace, objective, best_objective)
            best_success |= success
    return best.detach()


def evaluate_detector(detector, loader, attack_options, *, adaptive=False):
    clean_scores, adversarial_scores = [], []
    clean_correct, escaped_errors, total = 0, 0, 0
    options = dict(attack_options)
    method = options.pop("method")
    started = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device_of(detector.model)), labels.to(device_of(detector.model))
        if adaptive:
            adversarial = adaptive_attack(detector.model, detector, images, labels, **options)
        else:
            adversarial = attack(detector.model, images, labels, method, **options)
        with torch.no_grad():
            original = detector.model(images).argmax(1).eq(labels)
            scores = detector.score(adversarial)
            escaped = detector.model(adversarial).argmax(1).ne(labels) & (scores <= detector.threshold)
            clean_scores.extend(detector.score(images).tolist())
            adversarial_scores.extend(scores.tolist())
        clean_correct += int(original.sum())
        escaped_errors += int((escaped & original).sum())
        total += len(images)
    return dict(**detection_metrics(clean_scores, adversarial_scores, detector.threshold),
                attack=method, mode="adaptive" if adaptive else "transfer", parameters=attack_options,
                evasion_success_rate=escaped_errors / clean_correct if clean_correct else None,
                seconds=time.perf_counter() - started, samples=total)
