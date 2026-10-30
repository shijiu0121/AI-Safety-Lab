<script setup>
import { onMounted, onUnmounted, ref } from "vue";
import { apiRequest } from "../http";
const props = defineProps({ models: Array, health: Object });
const profiles = ref([]),
  datasets = ref([]),
  calibrations = ref([]),
  error = ref(""),
  notice = ref(""),
  busy = ref(false),
  editing = ref(""),
  preview = ref(""),
  previewFile = ref(null),
  importText = ref("");
const fresh = () => ({
  name: "",
  model_ids: [],
  aggregation: "soft_vote",
  transforms: [],
  detector: {
    method: "none",
    threshold: 0.5,
    min_input_std: 0,
    max_saturation: 1,
  },
  prediction_samples: 4,
  eot_samples: 4,
  seed: 0,
});
const form = ref(fresh());
const calibration = ref({
  defense_id: "",
  dataset_id: "",
  target_fpr: 0.05,
  sample_limit: 0,
  batch_size: 32,
});
const transformNames = {
  bit_depth: "位深压缩",
  median: "中值滤波",
  gaussian: "高斯滤波",
  resize_pad: "缩放与填充",
  noise: "高斯噪声",
};
const detectorNames = {
  none: "不启用输出检测",
  confidence: "低置信度",
  entropy: "预测熵",
  consistency: "滤波前后预测一致性",
};
let timer;
async function refresh() {
  [profiles.value, datasets.value, calibrations.value] = await Promise.all([
    apiRequest("/defenses"),
    apiRequest("/datasets"),
    apiRequest("/calibrations"),
  ]);
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
function add(kind) {
  form.value.transforms.push({
    kind,
    bits: 5,
    kernel: 3,
    sigma: 0.5,
    scale: 0.9,
    noise_std: 0.01,
  });
}
function edit(profile) {
  editing.value = profile.id;
  form.value = JSON.parse(JSON.stringify(profile.config));
  notice.value = "修改后会清除旧阈值校准，请重新检查和校准。";
  window.scrollTo(0, 0);
}
function reset() {
  editing.value = "";
  form.value = fresh();
}
function move(index, offset) {
  const list = form.value.transforms;
  const target = index + offset;
  if (target >= 0 && target < list.length)
    [list[index], list[target]] = [list[target], list[index]];
}
async function save() {
  await action(async () => {
    await apiRequest(
      editing.value ? `/defenses/${editing.value}` : "/defenses",
      {
        method: editing.value ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form.value),
      },
    );
    notice.value = "防御配置已保存。";
    reset();
  });
}
async function importProfile() {
  await action(async () => {
    await apiRequest("/defenses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(JSON.parse(importText.value)),
    });
    notice.value = "防御 JSON 已导入。";
  });
}
async function processPreview() {
  await action(async () => {
    if (!previewFile.value) throw new Error("先选择图片。");
    const payload = new FormData();
    payload.append("config", JSON.stringify(form.value.transforms));
    payload.append("image", previewFile.value);
    const response = await fetch("/api/transform-preview", {
      method: "POST",
      body: payload,
    });
    if (!response.ok) throw new Error((await response.json()).detail);
    if (preview.value) URL.revokeObjectURL(preview.value);
    preview.value = URL.createObjectURL(await response.blob());
  });
}
async function submitCalibration() {
  await action(async () => {
    await apiRequest("/calibrations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(calibration.value),
    });
    notice.value = "校准任务已加入队列。";
  });
}
const pct = (value) => (value == null ? "—" : (value * 100).toFixed(2) + "%");
onMounted(async () => {
  try {
    await refresh();
    timer = setInterval(
      () => refresh().catch((problem) => (error.value = problem.message)),
      4000,
    );
  } catch (problem) {
    error.value = problem.message;
  }
});
onUnmounted(() => {
  clearInterval(timer);
  if (preview.value) URL.revokeObjectURL(preview.value);
});
</script>
<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">DEFENSE STUDIO</span>
      <h1>防御与检测工作室</h1>
      <p>把输入变换、模型集成与检测规则组成一条可评测的流程。</p>
    </div>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <div v-if="notice" class="alert success">{{ notice }}</div>
  <form class="panel form-panel" @submit.prevent="save">
    <div class="panel-heading">
      <h2>{{ editing ? "编辑防御配置" : "创建防御流程" }}</h2>
      <button type="button" class="text-button" @click="reset">
        新建 / 重置
      </button>
    </div>
    <div class="form-grid">
      <label
        >名称<input
          v-model="form.name"
          required
          placeholder="例如：中值滤波＋预测一致性检测" /></label
      ><label
        >集成方式<select v-model="form.aggregation">
          <option value="soft_vote">概率平均 / 软投票</option>
          <option value="logit_mean">Logits 平均</option>
          <option value="hard_vote">多数投票</option>
          <option value="random_member">随机成员采样</option>
        </select></label
      >
    </div>
    <label>模型成员（顺序第一项是评测的基准模型）</label>
    <div class="checkbox-grid">
      <label v-for="model in models" :key="model.id"
        ><input type="checkbox" v-model="form.model_ids" :value="model.id" />{{
          model.name
        }}<small>{{ model.status }}</small></label
      >
    </div>
    <p v-if="!models.length" class="field-help">
      先到模型接入或资源管理登记模型；可以先登记配置，到服务器再验证。
    </p>
    <p class="field-help">
      集成成员要求输入大小、类别数量和类别顺序一致。当前成员顺序：{{
        form.model_ids
          .map((id) => models.find((model) => model.id === id)?.name)
          .join(" → ") || "未选择"
      }}。
    </p>
    <h3 class="section-subtitle">输入处理步骤 · 按顺序执行</h3>
    <div class="inline-actions wrap">
      <button
        v-for="(name, kind) in transformNames"
        :key="kind"
        type="button"
        class="button secondary small"
        :disabled="form.transforms.length >= 8"
        @click="add(kind)"
      >
        ＋ {{ name }}
      </button>
    </div>
    <div
      v-for="(step, index) in form.transforms"
      :key="index"
      class="pipeline-step"
    >
      <div class="inline-actions">
        <b>{{ index + 1 }}. {{ transformNames[step.kind] }}</b
        ><button class="text-button" type="button" @click="move(index, -1)">
          上移</button
        ><button class="text-button" type="button" @click="move(index, 1)">
          下移</button
        ><button
          class="text-button"
          type="button"
          @click="form.transforms.splice(index, 1)"
        >
          移除
        </button>
      </div>
      <div class="form-grid three-fields">
        <label v-if="step.kind === 'bit_depth'"
          >位数<input
            type="number"
            v-model.number="step.bits"
            min="1"
            max="8" /></label
        ><label v-if="['median', 'gaussian'].includes(step.kind)"
          >滤波核（奇数）<input
            type="number"
            v-model.number="step.kernel"
            min="1"
            max="9"
            step="2" /></label
        ><label v-if="step.kind === 'gaussian'"
          >σ<input
            type="number"
            v-model.number="step.sigma"
            min=".01"
            max="5"
            step="any" /></label
        ><label v-if="step.kind === 'resize_pad'"
          >缩放比例<input
            type="number"
            v-model.number="step.scale"
            min=".5"
            max="1"
            step="any" /></label
        ><label v-if="step.kind === 'noise'"
          >噪声标准差<input
            type="number"
            v-model.number="step.noise_std"
            min="0"
            max=".2"
            step="any"
        /></label>
      </div>
    </div>
    <p v-if="!form.transforms.length" class="field-help">
      当前直接使用原输入；可以只评测集成或输出检测。
    </p>
    <div class="form-grid three-fields">
      <label
        >检测方法<select v-model="form.detector.method">
          <option v-for="(name, key) in detectorNames" :key="key" :value="key">
            {{ name }}
          </option>
        </select></label
      ><label
        >检测阈值<input
          v-model.number="form.detector.threshold"
          type="number"
          min="0"
          max="1"
          step="any" /></label
      ><label
        >最小输入标准差<input
          v-model.number="form.detector.min_input_std"
          type="number"
          min="0"
          max=".5"
          step="any" /></label
      ><label
        >最大饱和像素比例<input
          v-model.number="form.detector.max_saturation"
          type="number"
          min="0"
          max="1"
          step="any" /></label
      ><label
        >随机预测采样数<input
          v-model.number="form.prediction_samples"
          type="number"
          min="1"
          max="32" /></label
      ><label
        >EOT 攻击采样数<input
          v-model.number="form.eot_samples"
          type="number"
          min="1"
          max="32" /></label
      ><label
        >随机种子<input v-model.number="form.seed" type="number" min="0"
      /></label>
    </div>
    <p class="field-help">
      选择模型成员、输入处理步骤和检测方法，保存后可进行阈值校准与攻防评测。
    </p>
    <button class="button" :disabled="busy || !form.model_ids.length">
      保存防御流程 →
    </button>
    <details class="advanced-options">
      <summary>输入处理预览</summary>
      <label
        >图片<input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          @change="previewFile = $event.target.files[0]" /></label
      ><button
        type="button"
        class="button secondary small"
        :disabled="busy"
        @click="processPreview"
      >
        执行当前处理步骤</button
      ><img
        v-if="preview"
        class="processing-preview"
        :src="preview"
        alt="实际输入处理预览"
      />
    </details>
  </form>
  <section class="panel module-section">
    <div class="panel-heading">
      <h2>已保存防御</h2>
      <span>{{ profiles.length }} PROFILES</span>
    </div>
    <div v-if="!profiles.length" class="empty-state">
      <span>◈</span><b>还没有防御配置</b>
      <p>保存一个流程后可以校准、评测或用于实时防护。</p>
    </div>
    <div class="profile-grid">
      <article
        v-for="profile in profiles"
        :key="profile.id"
        class="profile-card"
      >
        <b>{{ profile.name }}</b>
        <p>
          {{ profile.config.aggregation }} ·
          {{ profile.model_ids.length }} 个成员 ·
          {{ profile.config.transforms.length }} 个处理步骤
        </p>
        <p>
          {{ detectorNames[profile.config.detector.method] }} · 阈值
          {{ profile.config.detector.threshold }}
        </p>
        <span
          class="badge"
          :class="profile.calibration ? 'validated' : 'registered'"
          >{{ profile.calibration ? "已应用验证集阈值" : "尚未校准" }}</span
        >
        <p v-if="profile.calibration">
          {{ profile.calibration.sample_count }} 张验证图 · 经验误报率
          {{ pct(profile.calibration.empirical_fpr) }}
        </p>
        <div class="inline-actions wrap">
          <button class="text-button" @click="edit(profile)">编辑</button
          ><a :href="`/api/resources/defenses/${profile.id}/export`"
            >导出 JSON</a
          ><button
            class="text-button"
            :disabled="busy || !health?.compute_ready"
            @click="
              action(() =>
                apiRequest(`/defenses/${profile.id}/validate`, {
                  method: 'POST',
                }),
              )
            "
          >
            验证流程
          </button>
        </div>
      </article>
    </div>
  </section>
  <form
    class="panel form-panel module-section"
    @submit.prevent="submitCalibration"
  >
    <h2>检测阈值校准</h2>
    <div class="form-grid">
      <label
        >防御流程<select v-model="calibration.defense_id" required>
          <option disabled value="">选择防御</option>
          <option
            v-for="profile in profiles.filter(
              (item) => item.config.detector.method !== 'none',
            )"
            :key="profile.id"
            :value="profile.id"
          >
            {{ profile.name }}
          </option>
        </select></label
      ><label
        >独立验证集<select v-model="calibration.dataset_id" required>
          <option disabled value="">只显示已检查的 validation 划分</option>
          <option
            v-for="dataset in datasets.filter(
              (item) =>
                item.status === 'validated' && item.split === 'validation',
            )"
            :key="dataset.id"
            :value="dataset.id"
          >
            {{ dataset.name }}
          </option>
        </select></label
      ><label
        >目标误报率<input
          v-model.number="calibration.target_fpr"
          type="number"
          min=".001"
          max=".49"
          step="any" /></label
      ><label
        >样本数（0=完整验证集）<input
          v-model.number="calibration.sample_limit"
          type="number"
          min="0"
      /></label>
    </div>
    <p class="field-help">选择验证集与目标误报率，校准完成后点击应用阈值。</p>
    <button class="button" :disabled="busy || !health?.compute_ready">
      提交 GPU 校准
    </button>
    <div v-for="job in calibrations" :key="job.id" class="job-card">
      <b>{{ job.name }}</b
      ><span>{{ job.status }} · {{ job.progress }}%</span>
      <p v-if="job.error" class="field-error">{{ job.error }}</p>
      <p v-if="job.report">
        阈值 {{ job.report.threshold }} · 验证误报
        {{ pct(job.report.empirical_fpr) }} · {{ job.report.sample_count }} 张
      </p>
      <div class="inline-actions">
        <button
          v-if="job.status === 'succeeded'"
          type="button"
          class="text-button"
          @click="
            action(() =>
              apiRequest(`/calibrations/${job.id}/apply`, { method: 'POST' }),
            )
          "
        >
          应用该阈值</button
        ><button
          v-if="['queued', 'running'].includes(job.status)"
          type="button"
          class="text-button"
          @click="
            action(() =>
              apiRequest(`/workflows/calibrations/${job.id}/cancel`, {
                method: 'POST',
              }),
            )
          "
        >
          停止
        </button>
      </div>
    </div>
  </form>
  <details class="panel module-section advanced-options">
    <summary>导入防御 JSON</summary>
    <textarea
      v-model="importText"
      rows="8"
      placeholder="粘贴导出的防御配置；model_ids 需对应当前实例"
    ></textarea
    ><button class="button secondary small" @click="importProfile">
      导入配置
    </button>
  </details>
</template>
