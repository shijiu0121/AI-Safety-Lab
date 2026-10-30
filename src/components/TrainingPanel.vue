<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { apiRequest } from "../http";
const props = defineProps({ health: Object, models: Array });
const jobs = ref([]),
  selected = ref(null),
  telemetry = ref(null),
  error = ref(""),
  notice = ref(""),
  busy = ref(false),
  catalog = ref(null),
  name = ref(""),
  teacherId = ref(""),
  mean = ref("0.485,0.456,0.406"),
  std = ref("0.229,0.224,0.225"),
  jsonText = ref("");
const config = ref({
  dataset: "cifar10",
  data_dir: "./data/datasets",
  validation_dir: "",
  architecture: "cifar_resnet18",
  method: "pgd_at",
  epochs: 100,
  batch_size: 128,
  input_size: 32,
  learning_rate: 0.1,
  momentum: 0.9,
  weight_decay: 0.0005,
  epsilon_255: 8,
  attack_steps: 10,
  step_size_255: 2,
  validation_steps: 20,
  validation_fraction: 0.1,
  num_workers: 4,
  seed: 0,
  trades_beta: 6,
  download: false,
  imbalance_ratio: 1,
  augmentation: "standard",
  taet_warmup_epochs: 40,
  taet_alpha: 0.1,
  taet_beta: 0.1,
  taet_gamma: 0.1,
  temperature: 20,
  distill_alpha: 0.5,
  teacher_model_file: "",
});
const methods = {
  erm: "ERM 普通训练",
  pgd_at: "PGD 对抗训练",
  trades: "TRADES",
  at_bsl: "AT-BSL · 长尾对抗训练",
  taet: "TAET · 两阶段对抗均衡训练",
  distill_teacher: "防御蒸馏 · 温度教师训练",
  distill: "蒸馏 · 学生训练",
};
const metrics = computed(() => telemetry.value?.metrics || []);
const gpuReady = computed(
  () => props.health?.compute_ready && props.health?.device.startsWith("cuda"),
);
const paths = (file) =>
  `/api/workflows/training_runs/${selected.value.id}/files/${file}`;
const pct = (value) => (value == null ? "—" : (value * 100).toFixed(2) + "%");
let timer;
function configuration() {
  const result = JSON.parse(JSON.stringify(config.value));
  if (result.dataset === "imagefolder") {
    result.mean = mean.value.split(",").map(Number);
    result.std = std.value.split(",").map(Number);
  } else {
    result.validation_dir = "";
    delete result.mean;
    delete result.std;
  }
  return result;
}
async function refresh() {
  jobs.value = await apiRequest("/training-runs");
  if (selected.value) {
    telemetry.value = await apiRequest(
      `/training-runs/${selected.value.id}/telemetry`,
    );
    selected.value = telemetry.value.job;
  }
}
async function action(callback) {
  busy.value = true;
  error.value = "";
  notice.value = "";
  try {
    await callback();
    await refresh();
  } catch (problem) {
    error.value = problem.message;
  } finally {
    busy.value = false;
  }
}
async function save() {
  await action(async () => {
    selected.value = await apiRequest("/training-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: name.value || `${config.value.dataset} / ${config.value.method}`,
        teacher_model_id: teacherId.value,
        config: configuration(),
      }),
    });
    notice.value = "配置已保存。训练需要在 CUDA 服务器上显式启动。";
  });
}
async function open(job) {
  selected.value = job;
  await action(refresh);
}
function copy(job) {
  config.value = { ...job.config };
  name.value = job.name + " / 副本";
  teacherId.value = job.teacher_model_id || "";
  if (config.value.mean) mean.value = config.value.mean.join(",");
  if (config.value.std) std.value = config.value.std.join(",");
  delete config.value.output_dir;
  window.scrollTo(0, 0);
  notice.value = "已复制配置；保存后生成新的独立训练目录。";
}
function exportConfig() {
  try {
    const payload = configuration();
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "training-config.json";
    link.click();
    URL.revokeObjectURL(url);
  } catch (problem) {
    error.value = problem.message;
  }
}
function importConfig() {
  try {
    const payload = JSON.parse(jsonText.value);
    config.value = { ...config.value, ...payload };
    if (payload.mean) mean.value = payload.mean.join(",");
    if (payload.std) std.value = payload.std.join(",");
    notice.value = "已读取 JSON，保存时由后端校验全部字段。";
  } catch (problem) {
    error.value = problem.message;
  }
}
function curve(key, loss = false) {
  if (!metrics.value.length) return "";
  const max = loss
    ? Math.max(...metrics.value.map((item) => item[key]), 0.0001)
    : 1;
  const last = Math.max(...metrics.value.map((item) => item.epoch), 1);
  return metrics.value
    .map(
      (item) =>
        `${40 + ((item.epoch - 1) / Math.max(last - 1, 1)) * 490},${180 - (item[key] / max) * 150}`,
    )
    .join(" ");
}
onMounted(async () => {
  try {
    catalog.value = await apiRequest("/catalog");
    await refresh();
    timer = setInterval(
      () => refresh().catch((problem) => (error.value = problem.message)),
      3000,
    );
  } catch (problem) {
    error.value = problem.message;
  }
});
onUnmounted(() => clearInterval(timer));
</script>
<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">TRAINING CENTER</span>
      <h1>GPU 训练中心</h1>
      <p>配置训练方法与参数，查看任务日志、训练曲线和模型权重。</p>
    </div>
    <span class="phase-label">独立进程 · 统一队列</span>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <div v-if="notice" class="alert success">{{ notice }}</div>
  <form class="panel form-panel" @submit.prevent="save">
    <h2>准备训练任务</h2>
    <div class="form-grid">
      <label
        >任务名称<input
          v-model="name"
          placeholder="留空时使用数据集 / 方法" /></label
      ><label
        >训练方法<select v-model="config.method">
          <option v-for="(label, key) in methods" :key="key" :value="key">
            {{ label }}
          </option>
        </select></label
      ><label
        >数据类型<select v-model="config.dataset">
          <option value="cifar10">CIFAR-10</option>
          <option value="cifar100">CIFAR-100</option>
          <option value="imagefolder">ImageFolder</option>
        </select></label
      ><label
        >网络结构<select v-model="config.architecture">
          <option
            v-for="architecture in catalog?.architectures.filter(
              (item) => item !== 'custom',
            ) || []"
            :key="architecture"
          >
            {{ architecture }}
          </option>
        </select></label
      ><label>训练数据根目录<input v-model="config.data_dir" required /></label
      ><label v-if="config.dataset === 'imagefolder'"
        >独立验证目录（可留空后分层划分）<input v-model="config.validation_dir"
      /></label>
    </div>
    <div v-if="config.dataset === 'imagefolder'" class="form-grid">
      <label>RGB mean<input v-model="mean" required /></label
      ><label>RGB std<input v-model="std" required /></label>
    </div>
    <div class="form-grid three-fields">
      <label
        >输入大小<input
          v-model.number="config.input_size"
          type="number"
          min="16"
          max="512" /></label
      ><label
        >Epochs<input
          v-model.number="config.epochs"
          type="number"
          min="1"
          max="10000" /></label
      ><label
        >批大小<input
          v-model.number="config.batch_size"
          type="number"
          min="1"
          max="2048" /></label
      ><label
        >学习率<input
          v-model.number="config.learning_rate"
          type="number"
          min=".000001"
          max="10"
          step="any" /></label
      ><label
        >验证集比例<input
          v-model.number="config.validation_fraction"
          type="number"
          min=".001"
          max=".49"
          step="any" /></label
      ><label
        >训练不平衡率<input
          v-model.number="config.imbalance_ratio"
          type="number"
          min="1"
          max="10000"
      /></label>
    </div>
    <div v-if="['at_bsl', 'taet'].includes(config.method)" class="pipeline-step">
      <h3>长尾训练设置</h3>
      <p class="field-help">按训练不平衡率生成长尾数据，选择数据增强策略。</p>
      <label>数据增强<select v-model="config.augmentation"><option value="standard">标准裁剪与翻转</option><option value="randaugment">RandAugment (2, 8)</option><option value="autoaugment">AutoAugment · CIFAR-10 策略</option></select></label>
      <div v-if="config.method === 'taet'" class="form-grid">
        <label>第一阶段训练轮数<input v-model.number="config.taet_warmup_epochs" type="number" min="0" /></label>
        <label>均衡损失 α<input v-model.number="config.taet_alpha" type="number" min="0" step="any" /></label>
        <label>层次损失 β<input v-model.number="config.taet_beta" type="number" min="0" step="any" /></label>
        <label>类别损失 γ<input v-model.number="config.taet_gamma" type="number" min="0" step="any" /></label>
      </div>
    </div>
    <div
      v-if="['distill', 'distill_teacher'].includes(config.method)"
      class="pipeline-step"
    >
      <h3>蒸馏设置</h3>
      <p class="field-help">
        先训练温度教师，再使用教师软标签训练学生；也可以选择已有教师模型。
      </p>
      <div class="form-grid">
        <label
          >温度 T<input
            v-model.number="config.temperature"
            type="number"
            min=".01"
            max="1000"
            step="any" /></label
        ><label v-if="config.method === 'distill'"
          >硬标签损失占比 α<input
            v-model.number="config.distill_alpha"
            type="number"
            min="0"
            max="1"
            step="any" /></label
        ><label v-if="config.method === 'distill'"
          >已接入的教师（可选）<select v-model="teacherId">
            <option value="">使用下方服务器 JSON 路径</option>
            <option
              v-for="model in models.filter(
                (item) => item.status === 'validated',
              )"
              :key="model.id"
              :value="model.id"
            >
              {{ model.name }}
            </option>
          </select></label
        ><label v-if="config.method === 'distill' && !teacherId"
          >教师模型注册 JSON 路径<input
            v-model="config.teacher_model_file"
            required
            placeholder="data/training/teacher/model_registration.json"
        /></label>
      </div>
    </div>
    <details class="advanced-options">
      <summary>优化器、攻击与复现参数</summary>
      <div class="form-grid three-fields">
        <label
          >动量<input
            v-model.number="config.momentum"
            type="number"
            min="0"
            max=".999"
            step="any" /></label
        ><label
          >权重衰减<input
            v-model.number="config.weight_decay"
            type="number"
            min="0"
            max="1"
            step="any" /></label
        ><label
          >TRADES β<input
            v-model.number="config.trades_beta"
            type="number"
            min=".01"
            max="100"
            step="any" /></label
        ><label
          >训练预算（/255）<input
            v-model.number="config.epsilon_255"
            type="number"
            min="0"
            max="32" /></label
        ><label
          >训练攻击步数<input
            v-model.number="config.attack_steps"
            type="number"
            min="1"
            max="500" /></label
        ><label
          >攻击步长（/255）<input
            v-model.number="config.step_size_255"
            type="number"
            min=".001"
            max="32"
            step="any" /></label
        ><label
          >验证攻击步数<input
            v-model.number="config.validation_steps"
            type="number"
            min="1"
            max="500" /></label
        ><label
          >数据读取进程数<input
            v-model.number="config.num_workers"
            type="number"
            min="0"
            max="32" /></label
        ><label
          >随机种子<input v-model.number="config.seed" type="number" min="0"
        /></label>
      </div>
      <div class="checkbox-grid">
        <label
          ><input v-model="config.download" type="checkbox" />允许 TorchVision
          下载 CIFAR 数据（默认关闭）</label
        >
      </div>
    </details>
    <p class="field-help">
      训练任务自动分配输出目录，使用验证集选取最佳权重，并支持从最近完整 epoch
      恢复。
    </p>
    <div class="inline-actions wrap">
      <button class="button" :disabled="busy">保存训练配置 →</button
      ><button type="button" class="button secondary" @click="exportConfig">
        导出独立训练 JSON
      </button>
    </div>
    <details class="advanced-options">
      <summary>导入 / 高级 JSON</summary>
      <textarea
        v-model="jsonText"
        rows="10"
        placeholder="粘贴配置；未识别字段将由后端拒绝"
      ></textarea
      ><button
        type="button"
        class="button secondary small"
        @click="importConfig"
      >
        读取配置
      </button>
    </details>
  </form>
  <section class="panel module-section">
    <div class="panel-heading">
      <h2>训练任务</h2>
      <span>{{ jobs.length }} RUNS</span>
    </div>
    <p v-if="!jobs.length" class="empty-copy">暂无训练任务。</p>
    <div v-for="job in jobs" :key="job.id" class="job-card">
      <button class="text-button" @click="open(job)">{{ job.name }}</button
      ><span
        >{{ job.config.method }} · {{ job.status }} · {{ job.progress }}%</span
      ><button class="text-button" @click="copy(job)">复制配置</button>
    </div>
  </section>
  <section v-if="selected" class="panel module-section">
    <div class="panel-heading">
      <h2>{{ selected.name }}</h2>
      <span>{{ selected.status }}</span>
    </div>
    <p>{{ selected.message }}</p>
    <div v-if="selected.error" class="alert error">{{ selected.error }}</div>
    <div class="inline-actions wrap">
      <button
        v-if="selected.status === 'configured'"
        class="button small"
        :disabled="busy || !gpuReady"
        @click="
          action(() =>
            apiRequest(`/training-runs/${selected.id}/start`, {
              method: 'POST',
            }),
          )
        "
      >
        开始 GPU 训练</button
      ><button
        v-if="['failed', 'cancelled', 'interrupted'].includes(selected.status)"
        class="button small"
        :disabled="busy || !gpuReady || !telemetry?.checkpoint_available"
        @click="
          action(() =>
            apiRequest(`/training-runs/${selected.id}/resume`, {
              method: 'POST',
            }),
          )
        "
      >
        从完整 Epoch 恢复</button
      ><button
        v-if="['running', 'queued'].includes(selected.status)"
        class="button secondary small"
        @click="
          action(() =>
            apiRequest(`/workflows/training_runs/${selected.id}/cancel`, {
              method: 'POST',
            }),
          )
        "
      >
        请求停止</button
      ><button
        v-if="selected.status === 'succeeded'"
        class="button small"
        @click="
          action(async () => {
            await apiRequest(`/training-runs/${selected.id}/import-model`, {
              method: 'POST',
            });
            notice = '模型已接入，下一步执行模型验证。';
          })
        "
      >
        接入训练模型</button
      ><a
        v-if="selected.status === 'succeeded'"
        :href="paths('training/model_registration.json')"
        >模型注册 JSON</a
      ><a v-if="selected.status !== 'configured'" :href="paths('training.log')"
        >下载日志</a
      >
    </div>
    <p v-if="!gpuReady" class="field-help">
      当前只准备配置；训练入口要求可用 CUDA。
    </p>
    <p
      v-if="
        ['failed', 'cancelled', 'interrupted'].includes(selected.status) &&
        !telemetry?.checkpoint_available
      "
      class="field-help"
    >
      首个完整 epoch 前停止时尚无恢复点，请复制配置新建任务。
    </p>
    <div v-if="metrics.length" class="two-columns">
      <div>
        <h3 class="section-subtitle">训练损失</h3>
        <svg
          class="suite-chart"
          viewBox="0 0 560 210"
          role="img"
          aria-label="逐 Epoch 训练损失"
        >
          <path d="M40 20 V180 H530" fill="none" stroke="#d3e1d7" />
          <polyline
            :points="curve('training_loss', true)"
            fill="none"
            stroke="#7865b8"
            stroke-width="2.5"
          />
        </svg>
        <p class="field-help">
          最新损失
          {{
            metrics.at(-1).training_loss.toFixed(5)
          }}；纵轴按本任务损失范围缩放。
        </p>
      </div>
      <div>
        <h3 class="section-subtitle">验证集正常 / PGD 准确率</h3>
        <svg
          class="suite-chart"
          viewBox="0 0 560 210"
          role="img"
          aria-label="验证集正常和 PGD 准确率"
        >
          <path d="M40 20 V180 H530" fill="none" stroke="#d3e1d7" />
          <polyline
            :points="curve('validation_clean_accuracy')"
            fill="none"
            stroke="#419274"
            stroke-width="2.5"
          />
          <polyline
            :points="curve('validation_pgd_accuracy')"
            fill="none"
            stroke="#d0a15d"
            stroke-width="2.5"
          />
        </svg>
        <p class="field-help">
          绿色正常 {{ pct(metrics.at(-1).validation_clean_accuracy) }} / 金色
          PGD {{ pct(metrics.at(-1).validation_pgd_accuracy) }}，纵轴 0–100%。
        </p>
      </div>
    </div>
    <p v-else class="empty-copy">尚无完成 Epoch 的指标。</p>
    <pre class="runtime-log">{{ telemetry?.log || "进程尚未产生日志。" }}</pre>
    <details class="advanced-options">
      <summary>实际训练配置</summary>
      <pre>{{ JSON.stringify(selected.config, null, 2) }}</pre>
    </details>
  </section>
</template>
