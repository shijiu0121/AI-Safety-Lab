<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { methods, docs } from "./siteContent";
import ResultChart from "./components/ResultChart.vue";
import "./public.css";
const route = ref(location.hash.slice(2) || "home");
const update = () => {
  route.value = location.hash.slice(2) || "home";
};
window.addEventListener("hashchange", update);
onUnmounted(() => window.removeEventListener("hashchange", update));
const page = computed(() => route.value.split("/")[0]);
const activeMethod = computed(
  () => methods.find((m) => m.id === route.value.split("/")[1]) || methods[0],
);
const activeDoc = computed(
  () => docs.find((m) => m.id === route.value.split("/")[1]) || docs[0],
);
const nav = [
  ["home", "项目介绍"],
  ["methods", "方法与功能"],
  ["results", "实验成果"],
  ["models", "模型与下载"],
  ["docs", "使用文档"],
];
const records = ref([]),
  loadError = ref(""),
  copied = ref(false);
const dataUrl = (path) =>
  /^https?:\/\//.test(path) ? path : `${import.meta.env.BASE_URL}data/${path}`;
onMounted(async () => {
  try {
    const response = await fetch(dataUrl("results.json"));
    if (!response.ok) throw new Error(`成果数据加载失败 (${response.status})`);
    records.value = (await response.json()).experiments;
    const first = records.value.find(r => r.group === 'main') || records.value[0];
    if (first) {
      dataset.value = first.dataset;
      architecture.value = first.architecture;
      ratio.value = String(first.config.imbalance_ratio);
      seed.value = String(first.config.seed);
    }
  } catch (error) {
    loadError.value = error.message;
  }
});
const dataset = ref("all"),
  architecture = ref("all"),
  training = ref("all"),
  ratio = ref("all"),
  seed = ref("all");
const attack = ref("pgd20"),
  detector = ref("all"),
  mode = ref("transfer"),
  selected = ref("");
const resultTab = ref("robustness");
const classSeries = computed(() => filtered.value.map(row => ({
  name: label(row),
  points: (row.evaluation?.results?.find(r => r.attack === attack.value && r.parameters.epsilon_255 === 8)?.per_class || [])
    .map(r => ({x: r.class_id, y: r.robust_accuracy}))
})).filter(row => row.points.length));
const choices = (key) =>
  Array.from(
    new Set(
      records.value.map((row) =>
        key === "ratio"
          ? row.config.imbalance_ratio
          : key === "seed"
            ? row.config.seed
            : row[key],
      ),
    ),
  );
const filtered = computed(() =>
  records.value.filter(
    (row) =>
      (dataset.value === "all" || row.dataset === dataset.value) &&
      (architecture.value === "all" ||
        row.architecture === architecture.value) &&
      (training.value === "all" || row.method === training.value) &&
      (ratio.value === "all" ||
        String(row.config.imbalance_ratio) === ratio.value) &&
      (seed.value === "all" || String(row.config.seed) === seed.value),
  ),
);
const label = (row) =>
  `${row.dataset} · ${row.architecture} · ${row.method} · IR${row.config.imbalance_ratio} · s${row.config.seed}`;
const focused = computed(
  () =>
    filtered.value.find((row) => row.id === selected.value) ||
    filtered.value[0],
);
const attackChoices = computed(() =>
  Array.from(
    new Set(
      filtered.value.flatMap(
        (row) => row.evaluation?.results?.map((r) => r.attack) || [],
      ),
    ),
  ),
);
const robustSeries = computed(() =>
  filtered.value
    .map((row) => ({
      name: label(row),
      points: (row.evaluation?.results || [])
        .filter((r) => r.attack === attack.value)
        .map((r) => ({ x: r.parameters.epsilon_255, y: r.robust_accuracy }))
        .sort((a, b) => a.x - b.x),
    }))
    .filter((row) => row.points.length),
);
const detectionRows = computed(() =>
  filtered.value.flatMap((row) =>
    (row.detection || [])
      .filter((d) => detector.value === "all" || d.method === detector.value)
      .flatMap((d) =>
        d.results
          .filter(
            (r) =>
              r.mode === mode.value &&
              (mode.value === "adaptive" || r.attack === attack.value),
          )
          .map((r) => ({ ...r, model: label(row), detector: d.method })),
      ),
  ),
);
const rocSeries = computed(() =>
  detectionRows.value.map((r) => ({
    name: `${r.model} · ${r.detector}`,
    points: r.roc.map((p) => ({ x: p.fpr, y: p.tpr })),
  })),
);
const defenseRows = computed(() =>
  filtered.value.flatMap((row) =>
    (row.defenses || [])
      .filter((d) => d.mode === mode.value)
      .map((d) => ({ ...d, model: label(row) })),
  ),
);
const anomalyRows = computed(() =>
  filtered.value.flatMap((row) =>
    (row.anomaly?.results || []).map((r) => ({ ...r, model: label(row) })),
  ),
);
const cases = computed(() =>
  filtered.value.flatMap((row) =>
    (row.evaluation?.cases || [])
      .filter((c) => c.attack === attack.value)
      .map((c) => ({ ...c, model: label(row) })),
  ),
);
const historySeries = computed(() =>
  focused.value
    ? [
        {
          name: "干净验证准确率",
          points: focused.value.history.map((r) => ({
            x: r.epoch,
            y: r.validation_clean_accuracy,
          })),
        },
        {
          name: "PGD 验证准确率",
          points: focused.value.history.map((r) => ({
            x: r.epoch,
            y: r.validation_pgd_accuracy,
          })),
        },
      ]
    : [],
);
const lossSeries = computed(() =>
  focused.value
    ? [
        {
          name: "训练损失",
          points: focused.value.history.map((r) => ({
            x: r.epoch,
            y: r.training_loss,
          })),
        },
      ]
    : [],
);
const pct = (value) => (value == null ? "—" : `${(value * 100).toFixed(2)}%`);
async function copy(text) {
  await navigator.clipboard.writeText(text);
  copied.value = true;
  setTimeout(() => (copied.value = false), 1500);
}
const coverage = [
  ["8", "攻击配置"],
  ["7", "训练模式"],
  ["6", "攻击检测方法"],
  ["5", "输入变换"],
];
const supported = [
  "cifar_resnet18",
  "cifar_resnet34",
  "resnet18",
  "resnet34",
  "densenet121",
  "mobilenet_v2",
  "efficientnet_b0",
  "wide_resnet50_2",
];
</script>

<template>
  <div class="public-site">
    <header class="public-header">
      <a class="public-brand" href="#/home"
        ><span class="public-mark">a</span>Aegis <b>Lab</b><small>0.4</small></a
      >
      <nav aria-label="主导航">
        <a
          v-for="[id, title] in nav"
          :key="id"
          :href="`#/${id}`"
          :class="{ active: page === id }"
          >{{ title }}</a
        >
      </nav>
    </header>
    <main class="public-main">
      <template v-if="page === 'home'">
        <section class="public-hero">
          <div>
            <div class="public-eyebrow">
              <i></i> A PYTORCH TOOLKIT FOR ROBUSTNESS
            </div>
            <h1>理解攻击。<br />构建防御。<br /><em>用实验回答。</em></h1>
            <p>
              面向图像分类的 AI 安全算法库。<br />在同一套接口中，连接对抗攻击、鲁棒训练、<br
                class="desktop-break"
              />检测方法与可复现的实验结果。
            </p>
            <div class="public-actions">
              <a class="pub-button" href="#/docs/quickstart"
                >快速开始 <span>→</span></a
              ><a class="pub-button light" href="#/results"
                >查看实验成果 <span>↗</span></a
              >
            </div>
            <div class="hero-tags">
              PyTorch <span>·</span> Python API <span>·</span> GPU experiments
            </div>
          </div>
          <div class="library-visual">
            <div class="visual-heading">
              <span>YOUR MODEL. ONE INTERFACE.</span
              ><span class="terminal-dots">● ● ●</span>
            </div>
            <pre><span class="code-comment"># 在自己的项目中开始</span>
<span class="code-key">from</span> aegis_lab <span class="code-key">import</span> attack

adversarial = <span class="code-function">attack</span>(
    model, images, labels,
    method=<span class="code-string">"pgd100"</span>,
    epsilon_255=<span class="code-number">8</span>
)</pre>
            <div class="library-pipeline">
              <span>PyTorch 模型</span><b>→</b><span>攻击与防御</span><b>→</b
              ><span>实验成果</span>
            </div>
            <div class="visual-bottom">
              <span>import aegis_lab</span><span>v0.4.0</span>
            </div>
          </div>
        </section>
        <section class="coverage-strip">
          <div v-for="[number, name] in coverage" :key="name">
            <strong>{{ number }}</strong
            ><span>{{ name }}</span>
          </div>
        </section>
        <section class="public-section">
          <div class="section-heading">
            <div>
              <div class="public-eyebrow">THE LIBRARY</div>
              <h2>从一个函数，到一组完整实验。</h2>
            </div>
            <a href="#/methods">浏览全部方法 ↗</a>
          </div>
          <div class="method-overview">
            <a
              v-for="(method, i) in methods"
              :key="method.id"
              :href="`#/methods/${method.id}`"
              ><span class="method-index">0{{ i + 1 }}</span>
              <div>
                <h3>{{ method.name }}</h3>
                <p>{{ method.subtitle }}</p>
                <small>{{ method.items.join(" · ") }}</small>
              </div>
              <b>↗</b></a
            >
          </div>
        </section>
        <section class="experiment-intro">
          <div>
            <div class="public-eyebrow">EXPERIMENTS</div>
            <h2>同一套测试，<br />看清方法之间的差异。</h2>
            <p>
              切换模型与数据集，比较扰动预算曲线、检测
              ROC、长尾类别表现和代表样本。
            </p>
            <a href="#/results" class="text-link">打开成果总览 →</a>
          </div>
          <div class="experiment-map">
            <div>
              <span>常规分类 & 蒸馏</span
              ><strong>22 <small>训练任务</small></strong>
              <p>
                CIFAR-10 / CIFAR-100<br />ResNet-18 · DenseNet-121 · MobileNetV2
              </p>
            </div>
            <div>
              <span>长尾鲁棒性</span><strong>20 <small>训练任务</small></strong>
              <p>IR 10 / 100<br />ERM · PGD-AT · TRADES · AT-BSL · TAET</p>
            </div>
          </div>
        </section>
        <section v-if="records.length" class="public-section">
          <div class="section-heading">
            <h2>最近导出的实验</h2>
            <a href="#/models">查看模型与下载 →</a>
          </div>
          <div class="run-list">
            <a v-for="row in records.slice(0, 3)" :key="row.id" href="#/results"
              ><strong>{{ row.method }}</strong
              ><span>{{ row.dataset }} / {{ row.architecture }}</span
              ><span>IR {{ row.config.imbalance_ratio }} ↗</span></a
            >
          </div>
        </section>
      </template>

      <template v-else-if="page === 'methods'">
        <div class="page-intro">
          <div class="public-eyebrow">METHODS & CAPABILITIES</div>
          <h1>方法与功能</h1>
          <p>算法原理、公开接口与参数说明。</p>
        </div>
        <div class="reading-layout">
          <aside class="reading-nav">
            <a
              v-for="(method, i) in methods"
              :key="method.id"
              :href="`#/methods/${method.id}`"
              :class="{ active: activeMethod.id === method.id }"
              ><small>0{{ i + 1 }}</small
              >{{ method.name }}</a
            >
          </aside>
          <article class="reading-content">
            <div class="public-eyebrow">{{ activeMethod.subtitle }}</div>
            <h2>{{ activeMethod.name }}</h2>
            <p class="reading-lead">{{ activeMethod.description }}</p>
            <div class="method-tags">
              <span v-for="item in activeMethod.items" :key="item">{{
                item
              }}</span>
            </div>
            <h3>调用示例</h3>
            <div class="code-panel">
              <button @click="copy(activeMethod.code)">
                {{ copied ? "已复制" : "复制代码" }}
              </button>
              <pre>{{ activeMethod.code }}</pre>
            </div>
            <h3>参数</h3>
            <div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>参数</th>
                    <th>默认 / 示例</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="param in activeMethod.params" :key="param[0]">
                    <td>
                      <code>{{ param[0] }}</code>
                    </td>
                    <td>{{ param[1] }}</td>
                    <td>{{ param[2] }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <template v-if="activeMethod.refs.length"
              ><h3>论文与实现参考</h3>
              <div class="reference-links">
                <a
                  v-for="[name, url] in activeMethod.refs"
                  :key="url"
                  :href="url"
                  target="_blank"
                  rel="noreferrer"
                  >{{ name }} ↗</a
                >
              </div></template
            >
            <div class="next-links">
              <a href="#/results">查看相关实验成果 →</a
              ><a
                :href="`#/docs/${activeMethod.id === 'anomaly' ? 'detection' : activeMethod.id}`"
                >使用教程 ↗</a
              >
            </div>
          </article>
        </div>
      </template>

      <template v-else-if="page === 'results'">
        <div class="page-intro">
          <div class="public-eyebrow">RESULTS & COMPARISONS</div>
          <h1>实验成果</h1>
          <p>从模型表现到检测效果，逐项比较。</p>
        </div>
        <div class="result-summary">
          <div>
            <strong>{{ records.length }}</strong
            ><span>已导出模型</span>
          </div>
          <div>
            <strong>{{ methods.length }}</strong
            ><span>功能模块</span>
          </div>
          <div><strong>42</strong><span>计划训练任务</span></div>
          <a :href="dataUrl('experiment-plan.json')" download
            >下载完整实验配置 ↓</a
          >
        </div>
        <p v-if="loadError" role="alert" class="data-error">{{ loadError }}</p>
        <div class="result-filters">
          <label
            >数据集<select v-model="dataset">
              <option value="all">全部数据集</option>
              <option v-for="value in choices('dataset')" :key="value">
                {{ value }}
              </option>
            </select></label
          ><label
            >模型结构<select v-model="architecture">
              <option value="all">全部模型</option>
              <option v-for="value in choices('architecture')" :key="value">
                {{ value }}
              </option>
            </select></label
          ><label
            >训练方法<select v-model="training">
              <option value="all">全部方法</option>
              <option v-for="value in choices('method')" :key="value">
                {{ value }}
              </option>
            </select></label
          ><label
            >类别不平衡比<select v-model="ratio">
              <option value="all">全部 IR</option>
              <option
                v-for="value in choices('ratio')"
                :key="value"
                :value="String(value)"
              >
                {{ value }}
              </option>
            </select></label
          ><label
            >种子<select v-model="seed">
              <option value="all">全部种子</option>
              <option
                v-for="value in choices('seed')"
                :key="value"
                :value="String(value)"
              >
                {{ value }}
              </option>
            </select></label
          >
        </div>
        <div class="result-tabs">
          <button
            v-for="[id, title] in [
              ['robustness', '鲁棒性对比'],
              ['detection', '攻击检测'],
              ['defenses', '预处理与集成'],
              ['anomaly', '异常检测'],
              ['training', '训练曲线'],
              ['cases', '代表案例'],
            ]"
            :key="id"
            :class="{ active: resultTab === id }"
            @click="resultTab = id"
          >
            {{ title }}
          </button>
        </div>
        <div v-if="!filtered.length" class="empty-results">
          <span>◫</span>
          <h3>{{ records.length ? "没有匹配的实验" : "暂无实验结果" }}</h3>
          <p>
            {{
              records.length
                ? "调整上方筛选条件查看其他实验。"
                : "完成服务器实验后，导入结果即可查看曲线、案例和报告。"
            }}
          </p>
          <a href="#/docs/reproduce">查看实验复现流程 →</a>
        </div>
        <template v-else>
          <div class="comparison-controls">
            <label
              v-if="['robustness', 'detection', 'cases'].includes(resultTab)"
              >攻击方法
              <select v-model="attack">
                <option v-for="value in attackChoices" :key="value">
                  {{ value }}
                </option>
              </select></label
            ><label v-if="resultTab === 'detection'"
              >检测方法
              <select v-model="detector">
                <option value="all">全部方法</option>
                <option
                  v-for="value in [
                    'confidence',
                    'entropy',
                    'consistency',
                    'feature_squeezing',
                    'mahalanobis',
                    'layer_regression',
                  ]"
                  :key="value"
                >
                  {{ value }}
                </option>
              </select></label
            ><label v-if="['detection', 'defenses'].includes(resultTab)"
              >攻击模式
              <select v-model="mode">
                <option value="transfer">迁移攻击</option>
                <option value="adaptive">自适应攻击</option>
              </select></label
            >
          </div>
          <template v-if="resultTab === 'robustness'"
            ><div class="chart-panel">
              <h3>扰动预算与鲁棒准确率</h3>
              <ResultChart v-if="robustSeries.length" :series="robustSeries" />
              <details v-if="classSeries.length"><summary>逐类鲁棒准确率 · ε=8/255</summary><ResultChart :series="classSeries" x-label="类别 ID" y-label="逐类鲁棒准确率" /></details>
              <p v-else class="compact-empty">暂无该攻击的评测结果</p>
            </div>
            <div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>模型 / 方法</th>
                    <th>预算</th>
                    <th>干净准确率</th>
                    <th>鲁棒准确率</th>
                    <th>均衡鲁棒准确率</th>
                    <th>攻击成功率</th>
                  </tr>
                </thead>
                <tbody>
                  <template v-for="row in filtered" :key="row.id"
                    ><tr
                      v-for="metric in (row.evaluation?.results || []).filter(
                        (r) => r.attack === attack,
                      )"
                      :key="metric.parameters.epsilon_255"
                    >
                      <td>{{ label(row) }}</td>
                      <td>{{ metric.parameters.epsilon_255 }} / 255</td>
                      <td>{{ pct(metric.clean_accuracy) }}</td>
                      <td>{{ pct(metric.robust_accuracy) }}</td>
                      <td>{{ pct(metric.balanced_robust_accuracy) }}</td>
                      <td>{{ pct(metric.attack_success_rate) }}</td>
                    </tr></template
                  >
                </tbody>
              </table>
            </div></template
          >
          <template v-if="resultTab === 'detection'"
            ><div class="chart-panel">
              <h3>检测 ROC</h3>
              <ResultChart
                v-if="rocSeries.length"
                :series="rocSeries"
                x-label="假阳性率 FPR"
                y-label="真阳性率 TPR"
              />
              <p v-else class="compact-empty">暂无匹配的检测结果</p>
            </div>
            <div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>模型</th>
                    <th>检测方法</th>
                    <th>AUROC</th>
                    <th>TPR</th>
                    <th>FPR</th>
                    <th>规避成功率</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(r, i) in detectionRows" :key="i">
                    <td>{{ r.model }}</td>
                    <td>{{ r.detector }}</td>
                    <td>{{ pct(r.auroc) }}</td>
                    <td>{{ pct(r.tpr) }}</td>
                    <td>{{ pct(r.fpr) }}</td>
                    <td>{{ pct(r.evasion_success_rate) }}</td>
                  </tr>
                </tbody>
              </table>
            </div></template
          >
          <template v-if="resultTab === 'defenses'"
            ><div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>模型</th>
                    <th>策略</th>
                    <th>干净准确率</th>
                    <th>鲁棒准确率</th>
                    <th>攻击成功率</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(r, i) in defenseRows" :key="i">
                    <td>{{ r.model }}</td>
                    <td>{{ r.method }}</td>
                    <td>{{ pct(r.clean_accuracy) }}</td>
                    <td>{{ pct(r.robust_accuracy) }}</td>
                    <td>{{ pct(r.attack_success_rate) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p v-if="!defenseRows.length" class="compact-empty">
              暂无匹配的策略评估
            </p></template
          >
          <template v-if="resultTab === 'anomaly'"
            ><div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>模型</th>
                    <th>方法</th>
                    <th>异常类型</th>
                    <th>AUROC</th>
                    <th>TPR</th>
                    <th>FPR</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(r, i) in anomalyRows" :key="i">
                    <td>{{ r.model }}</td>
                    <td>{{ r.method }}</td>
                    <td>
                      {{ r.condition
                      }}{{ r.ood_dataset ? " / " + r.ood_dataset : "" }}
                    </td>
                    <td>{{ pct(r.auroc) }}</td>
                    <td>{{ pct(r.tpr) }}</td>
                    <td>{{ pct(r.fpr) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <details
              v-for="r in filtered.filter((x) => x.anomaly)"
              :key="r.id"
              class="experiment-detail"
            >
              <summary>{{ label(r) }} · 输出偏移窗口</summary>
              <pre>{{ JSON.stringify(r.anomaly.drift, null, 2) }}</pre>
            </details></template
          >
          <template v-if="resultTab === 'training'"
            ><label class="run-select"
              >选择训练记录
              <select v-model="selected">
                <option value="">选择模型</option>
                <option v-for="r in filtered" :key="r.id" :value="r.id">
                  {{ label(r) }}
                </option>
              </select></label
            >
            <div class="chart-panel">
              <h3>{{ focused?.method }} · 验证准确率</h3>
              <ResultChart
                :series="historySeries"
                x-label="Epoch"
                y-label="验证准确率"
              />
            </div>
            <div class="chart-panel">
              <h3>训练损失</h3>
              <ResultChart
                :series="lossSeries"
                x-label="Epoch"
                y-label="Loss"
                :percentage="false"
              /></div
          ></template>
          <template v-if="resultTab === 'cases'"
            ><div v-for="(sample, i) in cases" :key="i" class="case-card">
              <h3>{{ sample.model }}</h3>
              <p>
                {{ sample.attack }} · ε={{ sample.epsilon_255 }}/255 · 真实类别
                {{ sample.label }}
              </p>
              <div class="case-images">
                <figure>
                  <img :src="dataUrl(sample.original)" alt="原始图像" />
                  <figcaption>
                    原图 · 预测 {{ sample.clean_prediction }}
                  </figcaption>
                </figure>
                <figure>
                  <img :src="dataUrl(sample.adversarial)" alt="对抗图像" />
                  <figcaption>
                    对抗样本 · 预测 {{ sample.attacked_prediction }}
                  </figcaption>
                </figure>
                <figure>
                  <img :src="dataUrl(sample.difference)" alt="放大的扰动图" />
                  <figcaption>
                    扰动 × {{ sample.difference_scale }}，中心值 0.5
                  </figcaption>
                </figure>
              </div>
            </div>
            <p v-if="!cases.length" class="compact-empty">
              暂无该攻击的代表案例
            </p></template
          >
          <section class="public-section">
            <h2>配置与报告</h2>
            <details
              v-for="r in filtered"
              :key="r.id"
              class="experiment-detail"
            >
              <summary>{{ label(r) }}<span>展开参数与下载</span></summary>
              <pre>{{ r.command }}</pre>
              <pre>{{ JSON.stringify(r.config, null, 2) }}</pre>
              <div class="download-links">
                <a
                  v-for="(url, name) in r.downloads"
                  :key="name"
                  :href="dataUrl(url)"
                  download
                  >{{ name }} ↓</a
                >
              </div>
            </details>
          </section>
        </template>
      </template>

      <template v-else-if="page === 'models'"
        ><div class="page-intro">
          <div class="public-eyebrow">MODEL ZOO & DOWNLOADS</div>
          <h1>模型与下载</h1>
          <p>模型结构、训练配置、数据介绍与实验产物。</p>
        </div>
        <h2>支持的模型结构</h2>
        <div class="architecture-grid">
          <div v-for="name in supported" :key="name">
            <span>PyTorch</span>
            <h3>{{ name }}</h3>
            <small>{{
              name.startsWith("cifar_")
                ? "CIFAR 32 × 32 适配"
                : "torchvision 架构"
            }}</small>
          </div>
        </div>
        <div class="dataset-grid">
          <article>
            <span>DATASET / 01</span>
            <h3>CIFAR-10</h3>
            <p>
              10 类 · 32 × 32 彩色图像<br />50,000 张训练图像 / 10,000
              张测试图像
            </p>
            <p>训练集按类别划分训练与验证子集；支持 IR 10、100 长尾采样。</p>
          </article>
          <article>
            <span>DATASET / 02</span>
            <h3>CIFAR-100</h3>
            <p>
              100 类 · 32 × 32 彩色图像<br />50,000 张训练图像 / 10,000
              张测试图像
            </p>
            <p>使用细粒度类别标签，按类别报告正常和鲁棒准确率。</p>
          </article>
          <article>
            <span>CUSTOM DATA</span>
            <h3>ImageFolder</h3>
            <p>每个子目录对应一个类别。</p>
            <pre>dataset/\n  class_a/\n  class_b/</pre>
            <a href="#/docs/training">自定义训练 →</a>
          </article>
        </div>
        <section class="public-section">
          <div class="section-heading">
            <h2>模型卡与权重</h2>
            <a :href="dataUrl('experiment-plan.json')" download>实验配置 ↓</a>
          </div>
          <div v-if="!records.length" class="compact-empty">
            暂无已导出的模型权重
          </div>
          <div class="model-cards">
            <article v-for="r in records" :key="r.id">
              <span>{{ r.dataset }} / IR {{ r.config.imbalance_ratio }}</span>
              <h3>{{ r.architecture }}</h3>
              <p>
                {{ r.method }} · {{ r.config.epochs }} epochs · seed
                {{ r.config.seed }}
              </p>
              <dl>
                <dt>输入</dt>
                <dd>
                  {{ r.config.input_size }} × {{ r.config.input_size }} RGB
                </dd>
                <dt>训练预算</dt>
                <dd>{{ r.config.epsilon_255 }} / 255</dd>
              </dl>
              <div class="download-links">
                <a
                  v-if="r.downloads.weights"
                  :href="dataUrl(r.downloads.weights)"
                  download
                  >模型权重 ↓</a
                ><a
                  v-if="r.downloads['config.json']"
                  :href="dataUrl(r.downloads['config.json'])"
                  download
                  >配置 ↓</a
                ><a
                  v-if="r.downloads.report"
                  :href="dataUrl(r.downloads.report)"
                  target="_blank"
                  >实验报告 ↗</a
                >
              </div>
            </article>
          </div>
        </section></template
      >

      <template v-else-if="page === 'docs'"
        ><div class="page-intro">
          <div class="public-eyebrow">DOCUMENTATION</div>
          <h1>使用文档</h1>
          <p>安装、调用接口与完整实验复现。</p>
        </div>
        <div class="reading-layout">
          <aside class="reading-nav">
            <a
              v-for="doc in docs"
              :key="doc.id"
              :href="`#/docs/${doc.id}`"
              :class="{ active: activeDoc.id === doc.id }"
              >{{ doc.title }}</a
            >
          </aside>
          <article class="reading-content">
            <h2>{{ activeDoc.title }}</h2>
            <p class="reading-lead">{{ activeDoc.intro }}</p>
            <div class="code-panel">
              <button @click="copy(activeDoc.code)">
                {{ copied ? "已复制" : "复制代码" }}
              </button>
              <pre>{{ activeDoc.code }}</pre>
            </div>
            <p v-for="note in activeDoc.notes" :key="note" class="doc-note">
              {{ note }}
            </p>
            <div v-if="activeDoc.id === 'workbench'" class="public-actions">
              <a class="pub-button" href="#/workbench/overview" target="_blank"
                >打开本机工作台 ↗</a
              >
            </div>
            <div class="next-links">
              <a href="#/methods">方法与参数 →</a
              ><a :href="dataUrl('DSH_SERVER_HANDOFF.md')" download
                >服务器运行说明 ↓</a
              ><a :href="dataUrl('API.md')" download>完整 API 文档 ↓</a>
            </div>
          </article>
        </div></template
      >
      <template v-else
        ><div class="page-intro">
          <h1>页面未找到</h1>
          <a href="#/home">返回项目介绍 →</a>
        </div></template
      >
    </main>
    <footer class="public-footer">
      <a class="public-brand" href="#/home">Aegis <b>Lab</b></a
      ><span>对抗攻击 · 鲁棒学习 · 安全评估</span
      ><a href="#/docs/install">安装与使用 ↗</a>
    </footer>
  </div>
</template>
