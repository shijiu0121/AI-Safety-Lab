<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { apiRequest } from "../http";

const props = defineProps({
  models: { type: Array, required: true },
  health: { type: Object, default: null },
});
const datasets = ref([]),
  catalog = ref([]),
  jobs = ref([]),
  selectedJob = ref(null);
const error = ref(""),
  notice = ref(""),
  busy = ref(false),
  selectedCell = ref(null);
const graphModel = ref(""),
  graphNorm = ref("Linf");
const form = ref({
  dataset_id: "",
  model_ids: [],
  attacks: ["pgd20", "pgd100", "cw_linf", "autoattack", "deepfool_linf"],
  linf_budgets_255: "2,4,8",
  l2_budgets: "0.5",
  batch_size: 32,
  sample_limit: 100,
  seed: 0,
  step_size_255: 1,
  restarts: 1,
  cw_steps: 100,
  cw_confidence: 0,
  deepfool_steps: 50,
  deepfool_candidates: 10,
  deepfool_overshoot: 0.02,
});
const validModels = computed(() =>
  props.models.filter((model) => model.status === "validated"),
);
const validDatasets = computed(() =>
  datasets.value.filter((dataset) => dataset.status === "validated"),
);
const report = computed(() => selectedJob.value?.report);
const selectedResult = computed(() =>
  report.value?.cells.find((cell) => cell.id === selectedCell.value),
);
const statusNames = {
  queued: "排队中",
  running: "运行中",
  cancelling: "取消中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "服务中断",
};
let timer;

function budgets(value) {
  if (!value.trim()) return [];
  const tokens = value.split(",").map((item) => item.trim());
  if (tokens.some((item) => !item || !Number.isFinite(Number(item)))) return [];
  return tokens.map(Number);
}

function configuration() {
  return {
    ...form.value,
    linf_budgets_255: budgets(form.value.linf_budgets_255),
    l2_budgets: budgets(form.value.l2_budgets),
  };
}
function percent(value) {
  return value === null || value === undefined
    ? "—"
    : (value * 100).toFixed(2) + "%";
}
function budgetLabel(protocol) {
  return protocol.norm === "Linf"
    ? (protocol.budget * 255).toFixed(1).replace(/\.0$/, "") + "/255"
    : protocol.budget.toString();
}
const combinationCount = computed(
  () =>
    form.value.model_ids.length *
    form.value.attacks.reduce(
      (count, name) =>
        count +
        (catalog.value.find((item) => item.id === name)?.norm === "L2"
          ? budgets(form.value.l2_budgets).length
          : budgets(form.value.linf_budgets_255).length),
      0,
    ),
);
async function refresh() {
  [datasets.value, jobs.value] = await Promise.all([
    apiRequest("/datasets"),
    apiRequest("/batch-runs"),
  ]);
  if (selectedJob.value) {
    selectedJob.value = await apiRequest(`/batch-runs/${selectedJob.value.id}`);
    if (!graphModel.value && selectedJob.value.report?.cells.length)
      graphModel.value = selectedJob.value.report.cells[0].model_id;
  }
}
async function action(callback) {
  error.value = "";
  notice.value = "";
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
    selectedJob.value = await apiRequest("/batch-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configuration()),
    });
    notice.value = "任务已排队，进度按实际完成批次更新。";
  });
}
async function open(job) {
  await action(async () => {
    selectedJob.value = await apiRequest(`/batch-runs/${job.id}`);
    selectedCell.value = null;
    graphModel.value = "";
  });
}
async function cancel() {
  await action(() =>
    apiRequest(`/batch-runs/${selectedJob.value.id}/cancel`, {
      method: "POST",
    }),
  );
}
async function resume() {
  await action(() =>
    apiRequest(`/batch-runs/${selectedJob.value.id}/resume`, {
      method: "POST",
    }),
  );
}
function fileUrl(filename) {
  return `/api/batch-runs/${selectedJob.value.id}/files/${filename}`;
}
function exportConfiguration() {
  try {
    const dataset = datasets.value.find(
      (item) => item.id === form.value.dataset_id,
    );
    if (!dataset || !form.value.model_ids.length)
      throw new Error("请先选择已登记的数据集和模型。");
    const modelSpecs = props.models
      .filter((item) => form.value.model_ids.includes(item.id))
      .map(({ status, validation, created_at, error, ...model }) => model);
    // 模型定义的顺序与 model_ids 一致，命令行可直接检查；迁移后修正服务器路径。
    const ordered = form.value.model_ids.map((id) =>
      modelSpecs.find((model) => model.id === id),
    );
    if (
      !budgets(form.value.linf_budgets_255).length ||
      !budgets(form.value.l2_budgets).length
    )
      throw new Error("预算列表无效，请输入逗号分隔的数字。");
    const payload = { experiment: configuration(), dataset, models: ordered };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "evaluation-suite.json";
    link.click();
    URL.revokeObjectURL(url);
    notice.value =
      "已导出完整资源与实验配置。迁移后检查 root、weights_path 和 loader_path。";
  } catch (problem) {
    error.value = problem.message;
  }
}
const palette = [
  "#278b67",
  "#8170c2",
  "#d9a04f",
  "#4f8eaf",
  "#ba728a",
  "#809d54",
];
const chart = computed(() => {
  const cells = (report.value?.cells || []).filter(
    (cell) =>
      cell.model_id === graphModel.value &&
      cell.protocol.norm === graphNorm.value,
  );
  const values = [...new Set(cells.map((cell) => cell.protocol.budget))].sort(
    (a, b) => a - b,
  );
  const names = [...new Set(cells.map((cell) => cell.protocol.name))];
  // 横轴按真实预算间距绘制，只有一个预算时放在中间。
  const x = (budget) =>
    50 +
    (values.length <= 1
      ? 0.5
      : (budget - values[0]) / (values[values.length - 1] - values[0])) *
      500;
  const points = (cell) =>
    `${x(cell.protocol.budget)},${205 - cell.metrics.robust_accuracy * 170}`;
  return {
    values,
    names,
    series: names.map((name, index) => ({
      name,
      color: palette[index % palette.length],
      points: cells
        .filter((cell) => cell.protocol.name === name)
        .sort((a, b) => a.protocol.budget - b.protocol.budget)
        .map(points)
        .join(" "),
    })),
    x,
  };
});
onMounted(async () => {
  try {
    catalog.value = await apiRequest("/attacks");
    await refresh();
  } catch (problem) {
    error.value = problem.message;
  }
  timer = setInterval(() => {
    if (!busy.value && document.visibilityState === "visible")
      refresh().catch((problem) => (error.value = problem.message));
  }, 2500);
});
onUnmounted(() => clearInterval(timer));
</script>

<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">EVALUATION SUITE</span>
      <h1>批量安全评测</h1>
      <p>
        统一样本、模型和扰动预算，比较 PGD-20、PGD-100、CW、AA 与 DeepFool。
      </p>
    </div>
    <span class="phase-label">多攻击 · 多预算 · 可恢复</span>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <div v-if="notice" class="alert success" role="status">{{ notice }}</div>
  <form class="panel form-panel" @submit.prevent="submit">
    <div class="panel-heading">
      <h2>创建实验矩阵</h2>
      <span>{{ combinationCount }} 个模型 / 攻击 / 预算组合</span>
    </div>
    <div class="two-columns">
      <div>
        <label
          >评测数据集<select v-model="form.dataset_id" required>
            <option value="" disabled>选择已检查的数据集</option>
            <option
              v-for="dataset in validDatasets"
              :key="dataset.id"
              :value="dataset.id"
            >
              {{ dataset.name }} · {{ dataset.inspection.sample_count }} 张
            </option>
          </select></label
        >
        <p v-if="!validDatasets.length" class="field-help">
          <a href="#/workbench/datasets">先登记并检查数据集 →</a>
        </p>
        <label>待评测模型</label>
        <div class="checkbox-grid">
          <label v-for="model in validModels" :key="model.id"
            ><input
              type="checkbox"
              v-model="form.model_ids"
              :value="model.id"
            />{{ model.name }}</label
          >
          <p v-if="!validModels.length" class="field-help">
            暂无已验证模型。<a href="#/workbench/models">前往模型接入 →</a>
          </p>
        </div>
      </div>
      <div>
        <label>攻击方法</label>
        <div class="checkbox-grid attack-checkboxes">
          <label
            v-for="attack in catalog.filter((item) => item.id !== 'pgd')"
            :key="attack.id"
            ><input
              type="checkbox"
              v-model="form.attacks"
              :value="attack.id"
              :disabled="!attack.available"
            />{{ attack.name
            }}<span class="attack-tag">{{ attack.norm }}</span></label
          >
        </div>
        <p class="field-help">
          AA 使用 standard 配置；CW-L∞ 使用 margin 损失迭代攻击。
        </p>
      </div>
    </div>
    <div class="form-grid">
      <label
        >L∞ 预算列表（/255，逗号分隔）<input
          v-model="form.linf_budgets_255"
          required
          placeholder="2,4,8" /></label
      ><label
        >L2 预算列表（用于 DeepFool-L2）<input
          v-model="form.l2_budgets"
          required
          placeholder="0.25,0.5,1"
      /></label>
    </div>
    <div class="form-grid three-fields">
      <label
        >样本数（0 = 完整数据集）<input
          v-model.number="form.sample_limit"
          type="number"
          min="0"
          required /></label
      ><label
        >批大小<input
          v-model.number="form.batch_size"
          type="number"
          min="1"
          max="512"
          required /></label
      ><label
        >随机种子<input
          v-model.number="form.seed"
          type="number"
          min="0"
          required
      /></label>
    </div>
    <details class="advanced-options">
      <summary>高级协议参数</summary>
      <div class="form-grid three-fields">
        <label
          >PGD / CW 步长（/255）<input
            v-model.number="form.step_size_255"
            type="number"
            min="0.001"
            step="0.001"
            required /></label
        ><label
          >PGD / CW 重启次数<input
            v-model.number="form.restarts"
            type="number"
            min="1"
            max="20"
            required /></label
        ><label
          >CW 步数<input
            v-model.number="form.cw_steps"
            type="number"
            min="1"
            max="500"
            required /></label
        ><label
          >CW 置信参数 κ<input
            v-model.number="form.cw_confidence"
            type="number"
            min="0"
            step="0.1"
            required /></label
        ><label
          >DeepFool 最大步数<input
            v-model.number="form.deepfool_steps"
            type="number"
            min="1"
            max="500"
            required /></label
        ><label
          >DeepFool 候选类别数<input
            v-model.number="form.deepfool_candidates"
            type="number"
            min="2"
            max="1000"
            required
        /></label>
      </div>
      <p class="field-help">PGD-20 与 PGD-100 分别运行 20 步和 100 步。</p>
    </details>
    <div class="subtle-note">
      样本数设为 0 可评测完整数据集；使用相同配置可以恢复已完成的批次。
    </div>
    <div class="inline-actions suite-actions">
      <button
        class="button"
        type="submit"
        :disabled="
          busy ||
          !health?.compute_ready ||
          !form.model_ids.length ||
          !form.dataset_id ||
          !form.attacks.length
        "
      >
        提交 GPU 评测 →</button
      ><button
        class="button secondary"
        type="button"
        @click="exportConfiguration"
      >
        导出服务器配置
      </button>
    </div>
  </form>

  <section class="panel table-panel">
    <div class="panel-heading">
      <h2>批量任务</h2>
      <span>{{ jobs.length }} RUNS</span>
    </div>
    <div v-if="!jobs.length" class="empty-state compact">
      <span>▤</span><b>尚无批量实验</b>
      <p>提交评测后，在这里查看任务与结果。</p>
    </div>
    <table v-else>
      <thead>
        <tr>
          <th>数据集 / 模型</th>
          <th>攻击</th>
          <th>范围</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="job in jobs"
          :key="job.id"
          class="clickable"
          :class="{ highlighted: selectedJob?.id === job.id }"
          @click="open(job)"
        >
          <td>
            <b>{{ job.dataset_name }}</b
            ><small class="table-detail">{{
              job.model_names.join(" / ")
            }}</small>
          </td>
          <td>{{ job.config.attacks.join(" · ") }}</td>
          <td>{{ job.config.sample_limit || "完整数据集" }}</td>
          <td>
            <span class="badge" :class="job.status">{{
              statusNames[job.status]
            }}</span
            ><progress
              v-if="['queued', 'running'].includes(job.status)"
              :value="job.progress"
              max="100"
            ></progress>
          </td>
        </tr>
      </tbody>
    </table>
  </section>
  <section v-if="selectedJob" class="panel suite-results">
    <div class="panel-heading">
      <div>
        <h2>
          实验结果 <span class="mono">{{ selectedJob.id.slice(0, 8) }}</span>
        </h2>
        <small>{{ selectedJob.message }} · {{ selectedJob.progress }}%</small>
      </div>
      <div class="inline-actions">
        <button
          v-if="['queued', 'running'].includes(selectedJob.status)"
          class="button secondary small"
          @click="cancel"
        >
          取消任务</button
        ><button
          v-if="
            ['failed', 'cancelled', 'interrupted'].includes(selectedJob.status)
          "
          class="button secondary small"
          @click="resume"
        >
          检查并恢复
        </button>
      </div>
    </div>
    <div v-if="selectedJob.error" class="alert error">
      {{ selectedJob.error }}
    </div>
    <template v-if="report?.cells.length"
      ><div v-if="report.subset" class="alert warning">
        评测范围：{{ report.sample_count }} 张随机采样图片。
      </div>
      <div class="inline-actions report-downloads">
        <a :href="fileUrl('report.json')">JSON 报告 ↗</a
        ><template v-if="selectedJob.status === 'succeeded'"
          ><a :href="fileUrl('summary.csv')">汇总 CSV ↗</a
          ><a :href="fileUrl('per_class.csv')">逐类 CSV ↗</a
          ><a :href="fileUrl('samples.csv')">逐样本 CSV ↗</a></template
        ><a :href="fileUrl('manifest.json')">实验协议 ↗</a>
      </div>
      <div class="form-grid">
        <label
          >曲线对应模型<select v-model="graphModel">
            <option
              v-for="model in [
                ...new Map(
                  report.cells.map((cell) => [cell.model_id, cell]),
                ).values(),
              ]"
              :key="model.model_id"
              :value="model.model_id"
            >
              {{ model.model_name }}
            </option>
          </select></label
        ><label
          >曲线范数<select v-model="graphNorm">
            <option value="Linf">L∞</option>
            <option value="L2">L2</option>
          </select></label
        >
      </div>
      <svg
        class="suite-chart"
        viewBox="0 0 600 250"
        role="img"
        aria-label="相同模型与范数下，不同攻击的鲁棒准确率曲线"
      >
        <line
          v-for="fraction in [0, 0.5, 1]"
          :key="fraction"
          x1="50"
          :y1="205 - fraction * 170"
          x2="550"
          :y2="205 - fraction * 170"
          stroke="#e5ece7"
        />
        <text
          v-for="fraction in [0, 0.5, 1]"
          :key="'label' + fraction"
          x="7"
          :y="210 - fraction * 170"
        >
          {{ fraction * 100 }}%
        </text>
        <polyline
          v-for="series in chart.series"
          :key="series.name"
          :points="series.points"
          :stroke="series.color"
          fill="none"
          stroke-width="2.5"
        />
        <g v-for="series in chart.series" :key="'dots' + series.name">
          <circle
            v-for="point in series.points.split(' ').filter(Boolean)"
            :key="point"
            :cx="point.split(',')[0]"
            :cy="point.split(',')[1]"
            r="3.5"
            :fill="series.color"
          />
        </g>
        <text
          v-for="budget in chart.values"
          :key="budget"
          :x="chart.x(budget)"
          y="229"
          text-anchor="middle"
        >
          {{
            graphNorm === "Linf"
              ? Number((budget * 255).toFixed(3)) + "/255"
              : budget
          }}
        </text>
      </svg>
      <div class="chart-legend">
        <span v-for="series in chart.series" :key="series.name"
          ><i :style="{ background: series.color }"></i>{{ series.name }}</span
        ><span v-if="!chart.series.length">该模型与范数暂无结果。</span>
      </div>
      <div class="panel table-panel">
        <table class="batch-summary-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>攻击</th>
              <th>范数 / 预算</th>
              <th>正常准确率</th>
              <th>鲁棒准确率</th>
              <th>攻击成功率</th>
              <th>耗时</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="cell in report.cells"
              :key="cell.id"
              class="clickable"
              @click="selectedCell = cell.id"
            >
              <td>{{ cell.model_name }}</td>
              <td>{{ cell.protocol.name }}</td>
              <td>
                {{ cell.protocol.norm }} / {{ budgetLabel(cell.protocol) }}
              </td>
              <td>{{ percent(cell.metrics.clean_accuracy) }}</td>
              <td>{{ percent(cell.metrics.robust_accuracy) }}</td>
              <td>
                {{ percent(cell.metrics.attack_success_rate_on_clean_correct) }}
              </td>
              <td>{{ cell.seconds }} s</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h3 class="section-subtitle">同预算联合结果</h3>
      <div class="combined-cards">
        <article
          v-for="group in report.combined"
          :key="group.model_id + group.norm + group.budget"
        >
          <small
            >{{ group.model_name }} · {{ group.norm }} ·
            {{ budgetLabel(group) }}</small
          ><b>{{ percent(group.metrics.robust_accuracy) }}</b>
          <p>{{ group.attacks.join(" / ") }}</p>
        </article>
      </div>
      <p class="field-help">
        {{ report.note }}
        攻击成功率只在原本预测正确的样本上计算；耗时包含该组合的推理与攻击初始化。
      </p>
      <div v-if="selectedResult" class="cell-detail">
        <h3>{{ selectedResult.protocol.name }} · 逐类别与案例</h3>
        <div class="class-bars">
          <div
            v-for="item in selectedResult.metrics.per_class.filter(
              (item) => item.count,
            )"
            :key="item.class_index"
          >
            <span>类别 {{ item.class_index }} · {{ item.count }} 张</span>
            <div><i :style="{ width: percent(item.robust_accuracy) }"></i></div>
            <b>{{ percent(item.robust_accuracy) }}</b>
          </div>
        </div>
        <div class="case-gallery">
          <article
            v-for="example in selectedResult.examples"
            :key="example.sample_index"
          >
            <div>
              <img :src="fileUrl(example.original)" alt="原图" /><img
                :src="fileUrl(example.adversarial)"
                alt="攻击结果"
              />
            </div>
            <small
              >样本 {{ example.sample_index }} · 真值 {{ example.label }} · 原图
              / 攻击后</small
            >
          </article>
          <p v-if="!selectedResult.examples.length" class="field-help">
            本组合未保存成功攻击案例；请查看逐样本结果。
          </p>
        </div>
        <details class="protocol-details">
          <summary>查看实际协议</summary>
          <pre>{{ JSON.stringify(selectedResult.protocol, null, 2) }}</pre>
        </details>
      </div>
    </template>
    <div v-else class="subtle-note">等待评测结果。</div>
  </section>
</template>
