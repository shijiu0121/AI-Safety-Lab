<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { apiRequest } from "../http";
const props = defineProps({ health: Object });
const profiles = ref([]),
  datasets = ref([]),
  jobs = ref([]),
  selected = ref(null),
  cellId = ref(""),
  error = ref(""),
  busy = ref(false);
const form = ref({
  dataset_id: "",
  defense_ids: [],
  attacks: ["pgd20", "pgd100", "cw_linf"],
  modes: ["transfer", "adaptive"],
  linf_budgets_255: "8",
  l2_budgets: "0.5",
  sample_limit: 100,
  batch_size: 32,
  seed: 0,
  step_size_255: 1,
  restarts: 1,
  cw_steps: 100,
  cw_confidence: 0,
  evade_detector: true,
  evasion_weight: 1,
});
const report = computed(() => selected.value?.report),
  cell = computed(() =>
    report.value?.cells.find((item) => item.id === cellId.value),
  );
const attackNames = {
  fgsm: "FGSM",
  pgd20: "PGD-20",
  pgd100: "PGD-100",
  cw_linf: "CW-L∞",
  autoattack: "AA standard",
  deepfool_linf: "DeepFool-L∞",
  deepfool_l2: "DeepFool-L2",
};
const pct = (value) => (value == null ? "—" : (value * 100).toFixed(2) + "%");
const file = (name) =>
  `/api/workflows/defense_runs/${selected.value.id}/files/${name}`;
const values = (text) => {
  const tokens = text.split(",").map((item) => item.trim());
  if (tokens.some((item) => !item || !Number.isFinite(Number(item))))
    throw new Error("预算列表需要逗号分隔的有效数字，不能含空项。");
  return tokens.map(Number);
};
const configuration = () => ({
  ...form.value,
  linf_budgets_255: values(form.value.linf_budgets_255),
  l2_budgets: values(form.value.l2_budgets),
});
const rocPoints = computed(
  () =>
    cell.value?.roc?.points
      .map((point) => `${40 + point.fpr * 250},${270 - point.tpr * 240}`)
      .join(" ") || "",
);
let timer;
async function refresh() {
  [profiles.value, datasets.value, jobs.value] = await Promise.all([
    apiRequest("/defenses"),
    apiRequest("/datasets"),
    apiRequest("/defense-runs"),
  ]);
  if (selected.value)
    selected.value = await apiRequest(
      `/workflows/defense_runs/${selected.value.id}`,
    );
}
async function action(callback) {
  error.value = "";
  busy.value = true;
  try {
    await callback();
    await refresh();
  } catch (problem) {
    error.value = problem.message;
  } finally {
    busy.value = false;
  }
}
async function submit() {
  await action(async () => {
    selected.value = await apiRequest("/defense-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configuration()),
    });
    cellId.value = "";
  });
}
async function open(job) {
  await action(async () => {
    selected.value = await apiRequest(`/workflows/defense_runs/${job.id}`);
    cellId.value = "";
  });
}
async function exportConfiguration() {
  await action(async () => {
    const dataset = datasets.value.find(
      (item) => item.id === form.value.dataset_id,
    );
    if (!dataset || !form.value.defense_ids.length)
      throw new Error("先选择数据集与防御配置。");
    const defenses = profiles.value.filter((item) =>
      form.value.defense_ids.includes(item.id),
    );
    const allModels = await apiRequest("/models");
    const ids = new Set(defenses.flatMap((item) => item.model_ids));
    const payload = {
      experiment: configuration(),
      dataset,
      defenses: form.value.defense_ids.map((id) =>
        defenses.find((item) => item.id === id),
      ),
      models: allModels.filter((item) => ids.has(item.id)),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "defense-suite.json";
    link.click();
    URL.revokeObjectURL(url);
  });
}
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
onUnmounted(() => clearInterval(timer));
</script>
<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">ATTACK × DEFENSE</span>
      <h1>攻防对比评测</h1>
      <p>
        比较同批样本的模型基准、迁移攻击与自适应攻击，分别看鲁棒性和拒绝代价。
      </p>
    </div>
    <span class="phase-label">BPDA · EOT · 检测规避</span>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <form class="panel form-panel" @submit.prevent="submit">
    <h2>创建攻防实验</h2>
    <label
      >评测数据集<select v-model="form.dataset_id" required>
        <option disabled value="">选择已检查的数据集</option>
        <option
          v-for="dataset in datasets.filter(
            (item) => item.status === 'validated',
          )"
          :key="dataset.id"
          :value="dataset.id"
        >
          {{ dataset.name }} · {{ dataset.split }}
        </option>
      </select></label
    ><label>防御流程</label>
    <div class="checkbox-grid">
      <label v-for="profile in profiles" :key="profile.id"
        ><input
          type="checkbox"
          v-model="form.defense_ids"
          :value="profile.id"
        />{{ profile.name }}</label
      >
    </div>
    <p v-if="!profiles.length" class="field-help">
      先到防御工作室保存一个配置。
    </p>
    <label>攻击方法</label>
    <div class="checkbox-grid attack-checkboxes">
      <label v-for="(name, id) in attackNames" :key="id"
        ><input type="checkbox" v-model="form.attacks" :value="id" />{{
          name
        }}</label
      >
    </div>
    <div class="checkbox-grid">
      <label
        ><input
          type="checkbox"
          v-model="form.modes"
          value="transfer"
        />迁移：攻击首个成员，再送入防御</label
      ><label
        ><input
          type="checkbox"
          v-model="form.modes"
          value="adaptive"
        />自适应：针对完整防御流程攻击</label
      >
    </div>
    <p class="field-help">
      自适应评测使用 FGSM、PGD 或 CW。AA 与 DeepFool
      可用于迁移攻击及关闭检测规避后的确定性可微流程。
    </p>
    <div class="form-grid three-fields">
      <label
        >L∞ 预算（/255）<input
          v-model="form.linf_budgets_255"
          required /></label
      ><label>L2 预算<input v-model="form.l2_budgets" required /></label
      ><label
        >样本数（0=完整划分）<input
          type="number"
          v-model.number="form.sample_limit"
          min="0" /></label
      ><label
        >批大小<input
          type="number"
          v-model.number="form.batch_size"
          min="1"
          max="512" /></label
      ><label
        >随机种子<input
          type="number"
          v-model.number="form.seed"
          min="0" /></label
      ><label
        >PGD/CW 步长（/255）<input
          type="number"
          v-model.number="form.step_size_255"
          min=".001"
          max="32"
          step="any"
      /></label>
    </div>
    <details class="advanced-options">
      <summary>自适应攻击参数</summary>
      <div class="checkbox-grid">
        <label
          ><input
            v-model="form.evade_detector"
            type="checkbox"
          />同时规避检测</label
        >
      </div>
      <div class="form-grid three-fields">
        <label
          >规避惩罚系数<input
            v-model.number="form.evasion_weight"
            type="number"
            min="0"
            max="100"
            step="any" /></label
        ><label
          >重启次数<input
            v-model.number="form.restarts"
            type="number"
            min="1"
            max="20" /></label
        ><label
          >CW 步数<input
            v-model.number="form.cw_steps"
            type="number"
            min="1"
            max="500" /></label
        ><label
          >CW κ<input
            v-model.number="form.cw_confidence"
            type="number"
            min="0"
            max="100"
            step="any"
        /></label>
      </div>
    </details>
    <div class="inline-actions wrap">
      <button
        class="button"
        :disabled="
          busy ||
          !health?.compute_ready ||
          !form.defense_ids.length ||
          !form.modes.length
        "
      >
        提交 GPU 攻防评测 →</button
      ><button
        class="button secondary"
        type="button"
        @click="exportConfiguration"
      >
        导出服务器配置
      </button>
    </div>
  </form>
  <section class="panel module-section">
    <div class="panel-heading">
      <h2>攻防任务</h2>
      <span>{{ jobs.length }} RUNS</span>
    </div>
    <p v-if="!jobs.length" class="empty-copy">暂无攻防实验。</p>
    <button
      v-for="job in jobs"
      :key="job.id"
      class="job-card job-button"
      @click="open(job)"
    >
      <b>{{ job.name }}</b
      ><span
        >{{ job.dataset_name }} · {{ job.status }} · {{ job.progress }}%</span
      >
    </button>
  </section>
  <section v-if="selected" class="panel module-section">
    <div class="panel-heading">
      <h2>实验结果 {{ selected.id.slice(0, 8) }}</h2>
      <div class="inline-actions">
        <button
          v-if="['queued', 'running'].includes(selected.status)"
          class="text-button"
          @click="
            action(() =>
              apiRequest(`/workflows/defense_runs/${selected.id}/cancel`, {
                method: 'POST',
              }),
            )
          "
        >
          停止</button
        ><button
          v-if="
            ['failed', 'cancelled', 'interrupted'].includes(selected.status)
          "
          class="text-button"
          @click="
            action(() =>
              apiRequest(`/workflows/defense_runs/${selected.id}/resume`, {
                method: 'POST',
              }),
            )
          "
        >
          检查并恢复
        </button>
      </div>
    </div>
    <p>{{ selected.message }} · {{ selected.progress }}%</p>
    <progress
      v-if="['running', 'queued'].includes(selected.status)"
      max="100"
      :value="selected.progress"
    ></progress>
    <div v-if="selected.error" class="alert error">{{ selected.error }}</div>
    <template v-if="report?.cells.length"
      ><div v-if="report.subset" class="alert warning">
        {{ report.sample_count }} 张随机子集，尚非完整测试集。
      </div>
      <div class="inline-actions report-downloads">
        <a :href="file('report.json')">JSON</a
        ><a :href="file('manifest.json')">实验协议</a
        ><template v-if="selected.status === 'succeeded'"
          ><a :href="file('report.html')">HTML 报告</a
          ><a :href="file('summary.csv')">汇总 CSV</a
          ><a :href="file('per_class.csv')">逐类 CSV</a
          ><a :href="file('samples.csv')">逐样本 CSV</a></template
        >
      </div>
      <div class="table-scroll">
        <table class="batch-summary-table">
          <thead>
            <tr>
              <th>防御 / 攻击 / 预算</th>
              <th>基准鲁棒率</th>
              <th>防御正常率</th>
              <th>防御鲁棒率</th>
              <th>正常告警率</th>
              <th>攻击覆盖率</th>
              <th>未检出错误率</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="item in report.cells"
              :key="item.id"
              class="clickable"
              :class="{ highlighted: cellId === item.id }"
              @click="cellId = item.id"
            >
              <td>
                <b>{{ item.defense_name }}</b
                ><small class="table-detail"
                  >{{ item.protocol.name }} · {{ item.protocol.norm }} ε={{
                    item.protocol.norm === "Linf"
                      ? Number((item.protocol.budget * 255).toFixed(3)) + "/255"
                      : item.protocol.budget
                  }}</small
                >
              </td>
              <td>{{ pct(item.baseline.robust_accuracy) }}</td>
              <td>{{ pct(item.metrics.clean_accuracy) }}</td>
              <td>{{ pct(item.metrics.robust_accuracy) }}</td>
              <td>{{ pct(item.detection.clean_flag_rate) }}</td>
              <td>{{ pct(item.detection.adversarial_coverage) }}</td>
              <td>{{ pct(item.detection.undetected_error_rate) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h3 class="section-subtitle">鲁棒性与防御代价</h3>
      <div
        v-for="item in report.cells"
        :key="'bar' + item.id"
        class="metric-bar"
      >
        <span>{{ item.defense_name }} / {{ item.protocol.name }}</span
        ><meter min="0" max="1" :value="item.metrics.robust_accuracy"></meter
        ><b>{{ pct(item.metrics.robust_accuracy) }}</b>
      </div>
      <h3 class="section-subtitle">同预算多攻击联合鲁棒率</h3>
      <div class="combined-cards">
        <article
          v-for="group in report.combined"
          :key="
            JSON.stringify([
              group.defense_id,
              group.mode,
              group.norm,
              group.budget,
            ])
          "
        >
          <small
            >{{ profiles.find((item) => item.id === group.defense_id)?.name }} ·
            {{ group.mode }} · {{ group.norm }} ε={{
              group.norm === "Linf"
                ? Number((group.budget * 255).toFixed(3)) + "/255"
                : group.budget
            }}</small
          ><b>{{ pct(group.metrics.robust_accuracy) }}</b>
          <p>{{ group.attacks.join(" / ") }}</p>
        </article>
      </div>
      <p class="field-help">点击结果行查看逐类别指标、检测分数 ROC 和案例。</p>
      <div v-if="cell" class="cell-detail">
        <h3>{{ cell.defense_name }} · {{ cell.protocol.name }}</h3>
        <div class="two-columns">
          <div>
            <h3>逐类别鲁棒率</h3>
            <div class="class-bars">
              <div
                v-for="category in cell.metrics.per_class"
                :key="category.class_index"
              >
                <span
                  >{{ report.classes[category.class_index] }} ·
                  {{ category.count }} 张</span
                >
                <div>
                  <i
                    :style="{
                      width: (category.robust_accuracy || 0) * 100 + '%',
                    }"
                  ></i>
                </div>
                <b>{{ pct(category.robust_accuracy) }}</b>
              </div>
            </div>
          </div>
          <div>
            <h3>
              检测分数 ROC · AUC
              {{ cell.roc.auc == null ? "—" : cell.roc.auc.toFixed(3) }}
            </h3>
            <svg
              class="roc-chart"
              viewBox="0 0 320 310"
              role="img"
              aria-label="检测分数经验 ROC 曲线"
            >
              <path d="M40 30 V270 H290" fill="none" stroke="#b5cabc" />
              <path
                d="M40 270 L290 30"
                stroke="#d9e5de"
                stroke-dasharray="4 4"
              />
              <polyline
                :points="rocPoints"
                fill="none"
                stroke="#29876a"
                stroke-width="2.5"
              />
              <text x="120" y="300">正常图误报率</text>
              <text x="4" y="20">扰动图检出率</text>
            </svg>
          </div>
        </div>
        <div class="case-gallery">
          <article v-for="example in cell.examples" :key="example.sample_index">
            <div class="triple">
              <img :src="file(example.original)" alt="原图" /><img
                :src="file(example.adversarial)"
                alt="攻击图"
              /><img :src="file(example.processed)" alt="处理图预览" />
            </div>
            <small
              >样本 {{ example.sample_index }} · 原图 / 攻击 / 处理预览</small
            >
          </article>
        </div>
        <details class="advanced-options">
          <summary>查看防御与实际攻击协议</summary>
          <pre>{{
            JSON.stringify(
              {
                defense: cell.defense_protocol,
                attack: cell.actual_attack_protocol,
                detection: cell.detection,
              },
              null,
              2,
            )
          }}</pre>
        </details>
      </div></template
    >
  </section>
</template>
