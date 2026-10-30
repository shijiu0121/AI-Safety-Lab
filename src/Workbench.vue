<script setup>
import { useLab } from "./useLab";
import DatasetPanel from "./components/DatasetPanel.vue";
import BatchPanel from "./components/BatchPanel.vue";
import SingleAttackPanel from "./components/SingleAttackPanel.vue";
import DefensePanel from "./components/DefensePanel.vue";
import DefenseEvaluationPanel from "./components/DefenseEvaluationPanel.vue";
import TrainingPanel from "./components/TrainingPanel.vue";
import GuardPanel from "./components/GuardPanel.vue";
import ResourcePanel from "./components/ResourcePanel.vue";
const {
  attackCatalog,
  attackName,
  budgetLabel,
  page,
  health,
  models,
  experiments,
  error,
  notice,
  busy,
  selectedExperimentId,
  modelDialogOpen,
  imagePreview,
  navigation,
  titles,
  validModels,
  runningCount,
  completedCount,
  selectedExperiment,
  result,
  experimentForm,
  modelForm,
  navigate,
  formatDate,
  percentage,
  statusName,
  registerModel,
  importCachedExample,
  validateModel,
  chooseArchitecture,
  chooseImage,
  startExperiment,
  openExperiment,
  cancelExperiment,
  imageUrl,
  tracePoints,
} = useLab();
</script>

<template>
  <div v-if="page === 'home'" class="site">
    <header class="site-header">
      <a class="brand" href="#/home"
        ><span class="brand-symbol"><i></i><i></i><i></i><i></i></span>Aegis
        <b>Lab</b></a
      >
      <nav>
        <button class="button small" @click="navigate('overview')">
          进入工作台 <span>↗</span>
        </button>
      </nav>
    </header>
    <main class="landing">
      <section class="hero">
        <div class="hero-copy">
          <div class="eyebrow">
            <span class="status-dot"></span> SELF-HOSTED · PYTORCH · CUDA
          </div>
          <h1>接入你的模型，<br /><span>开始安全评测。</span></h1>
          <p>
            从模型兼容性检查，到对抗攻击与结果分析。<br />在你自己的服务器上，完成一次可追溯的安全实验。
          </p>
          <div class="hero-caption">你的模型 · 你的 GPU · 你的实验记录</div>
        </div>
        <div class="hero-visual" aria-label="模型、攻击和报告的工作流程示意">
          <div class="visual-top">
            <span>EXPERIMENT WORKFLOW</span><span>03 / ATTACK & DEFENSE</span>
          </div>
          <div class="visual-flow">
            <div class="visual-node">
              <span class="node-icon">◇</span><b>MODEL</b
              ><small>模型与预处理</small>
            </div>
            <span class="flow-arrow">→</span>
            <div class="visual-node accented">
              <span class="node-icon">⌁</span><b>ATTACK</b
              ><small>生成对抗样本</small>
            </div>
            <span class="flow-arrow">→</span>
            <div class="visual-node">
              <span class="node-icon">▤</span><b>REPORT</b
              ><small>配置与结果留存</small>
            </div>
          </div>
          <div class="visual-code">
            <span class="code-dim"># 从明确的输入开始</span><br /><span
              class="code-green"
              >model</span
            >
            → validation<br /><span class="code-purple">attack</span> → PGD / CW
            / AA / DeepFool<br /><span class="code-green">result</span> → images
            + metrics + trace
          </div>
        </div>
      </section>
      <section class="feature-strip">
        <article>
          <span class="feature-number">01</span>
          <h3>带着自己的模型来</h3>
          <p>
            接入已有结构与权重，或提供自定义加载函数。明确输入范围、归一化与类别顺序。
          </p>
        </article>
        <article>
          <span class="feature-number">02</span>
          <h3>让攻击过程可见</h3>
          <p>
            选择 PGD、CW、AA 或 DeepFool，比较多模型、多预算的鲁棒性与成功案例。
          </p>
        </article>
        <article>
          <span class="feature-number">03</span>
          <h3>每次实验都有记录</h3>
          <p>
            保存模型摘要、输入、参数与运行环境。导出 JSON
            报告与浮点结果，继续分析。
          </p>
        </article>
      </section>
      <section class="feature-strip" aria-label="鲁棒性训练方法">
        <article>
          <span class="feature-number">ROBUST TRAINING</span>
          <h3>PGD-AT 与 TRADES</h3>
          <p>使用对抗样本训练模型，比较正常准确率与鲁棒准确率。</p>
        </article>
        <article>
          <span class="feature-number">LONG-TAILED</span>
          <h3>AT-BSL</h3>
          <p>结合对抗训练与平衡 Softmax，支持 RandAugment 和 AutoAugment。</p>
        </article>
        <article>
          <span class="feature-number">TWO-STAGE</span>
          <h3>TAET</h3>
          <p>通过两阶段训练与类别均衡损失研究长尾数据的模型鲁棒性。</p>
        </article>
      </section>
    </main>
    <footer class="site-footer">
      <span>Aegis Lab / 模型安全实验室</span
      ><span>独立部署 · 开放接口 · 可追溯实验</span>
    </footer>
  </div>

  <div v-else class="workspace">
    <aside class="sidebar">
      <a class="brand" href="#/home"
        ><span class="brand-symbol"><i></i><i></i><i></i><i></i></span>Aegis
        <b>Lab</b></a
      >
      <div class="sidebar-label">EXPERIMENT WORKSPACE</div>
      <nav>
        <a
          v-for="item in navigation"
          :key="item.id"
          :href="'#/workbench/' + item.id"
          :class="{ active: page === item.id }"
          ><span>{{ item.icon }}</span
          >{{ item.label
          }}<small v-if="item.id === 'experiments' && runningCount">{{
            runningCount
          }}</small></a
        >
      </nav>
      <div class="sidebar-bottom">
        <span
          class="status-dot"
          :class="{ warning: !health?.compute_ready }"
        ></span>
        <div>
          {{ health?.compute_ready ? "计算服务就绪" : "GPU 未连接"
          }}<small>{{
            health?.gpu_name || (health?.device === "cpu" ? "CPU" : "CUDA")
          }}</small>
        </div>
      </div>
    </aside>
    <section class="work-area">
      <header class="work-header">
        <span
          ><span class="muted">实验工作台</span><span class="separator">/</span
          >{{ titles[page] || "工作台" }}</span
        >
        <div>
          <a href="#/home">项目首页 ↗</a>
        </div>
      </header>
      <main class="work-content">
        <div v-if="error" class="alert error" role="alert">
          {{ error
          }}<button @click="error = ''" aria-label="关闭错误提示">×</button>
        </div>
        <div v-if="notice" class="alert success" role="status">
          {{ notice
          }}<button @click="notice = ''" aria-label="关闭提示">×</button>
        </div>

        <template v-if="page === 'overview'">
          <div class="page-heading">
            <div>
              <span class="eyebrow">OVERVIEW</span>
              <h1>开始一次模型安全实验</h1>
              <p>接入模型、配置攻击、训练防御模型并查看评测结果。</p>
            </div>
            <button class="button" @click="navigate('models')">
              接入模型 <span>＋</span>
            </button>
          </div>
          <div class="stat-grid">
            <article>
              <span>已接入模型</span><strong>{{ models.length }}</strong
              ><small>{{ validModels.length }} 个通过兼容性检查</small>
            </article>
            <article>
              <span>已完成任务</span><strong>{{ completedCount }}</strong
              ><small>查看任务结果与报告</small>
            </article>
            <article>
              <span>正在执行 / 排队</span><strong>{{ runningCount }}</strong
              ><small>查看任务队列</small>
            </article>
            <article>
              <span>计算设备</span
              ><strong class="device-text">{{
                health?.device?.toUpperCase() || "—"
              }}</strong
              ><small>{{
                health?.gpu_name || (health ? "未检测到可用 GPU" : "服务未连接")
              }}</small>
            </article>
          </div>
          <div class="two-columns">
            <section class="panel">
              <div class="panel-heading">
                <h2>实验路径</h2>
                <span>GET STARTED</span>
              </div>
              <button class="step-row" @click="navigate('models')">
                <span class="step-number">1</span>
                <div>
                  <b>接入并验证模型</b>
                  <p>检查权重、预处理、输出和梯度。</p>
                </div>
                <span>→</span></button
              ><button class="step-row" @click="navigate('attack')">
                <span class="step-number">2</span>
                <div>
                  <b>配置对抗攻击</b>
                  <p>上传图片，选择攻击和扰动预算。</p>
                </div>
                <span>→</span></button
              ><button class="step-row" @click="navigate('experiments')">
                <span class="step-number">3</span>
                <div>
                  <b>检查结果与运行记录</b>
                  <p>比较预测，核对扰动，导出报告。</p>
                </div>
                <span>→</span>
              </button>
            </section>
            <section class="panel">
              <div class="panel-heading">
                <h2>实例信息</h2>
                <span>RUNTIME</span>
              </div>
              <dl class="definition-list">
                <dt>PyTorch</dt>
                <dd>{{ health?.torch || "—" }}</dd>
                <dt>TorchVision</dt>
                <dd>{{ health?.torchvision || "—" }}</dd>
                <dt>默认设备</dt>
                <dd>{{ health?.device || "cuda" }}</dd>
                <dt>运行方式</dt>
                <dd>用户独立部署</dd>
                <dt>数据存储</dt>
                <dd>本实例 SQLite 与文件目录</dd>
                <dt>攻击方法</dt>
                <dd>PGD-20 / PGD-100 / CW / AA / DeepFool</dd>
              </dl>
            </section>
          </div>
          <section class="panel table-panel">
            <div class="panel-heading">
              <h2>最近单图实验</h2>
              <a href="#/workbench/experiments">全部记录 →</a>
            </div>
            <div v-if="!experiments.length" class="empty-state compact">
              <span>▤</span><b>还没有单图实验记录</b>
              <p>接入一个模型后，从攻击实验室开始。</p>
            </div>
            <table v-else>
              <thead>
                <tr>
                  <th>模型</th>
                  <th>攻击</th>
                  <th>预算</th>
                  <th>状态</th>
                  <th>创建时间</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="task in experiments.slice(0, 5)"
                  :key="task.id"
                  @click="openExperiment(task)"
                  class="clickable"
                >
                  <td>{{ task.model_name }}</td>
                  <td>{{ attackName(task) }}</td>
                  <td>{{ budgetLabel(task) }}</td>
                  <td>
                    <span class="badge" :class="task.status">{{
                      statusName(task.status)
                    }}</span>
                  </td>
                  <td>{{ formatDate(task.created_at) }}</td>
                </tr>
              </tbody>
            </table>
          </section>
        </template>

        <template v-if="page === 'models'">
          <div class="page-heading">
            <div>
              <span class="eyebrow">MODEL REGISTRY</span>
              <h1>接入你的 PyTorch 模型</h1>
              <p>登记模型结构、权重路径和预处理参数，检查模型兼容性。</p>
            </div>
            <button class="button" @click="modelDialogOpen = true">
              新增模型 <span>＋</span>
            </button>
          </div>
          <div class="info-banner">
            <span>◇</span>
            <div>
              <b>模型配置</b>
              <p>
                请同时提供网络结构、输入尺寸、归一化和类别顺序。自定义模型通过本机
                Python 加载函数接入。
              </p>
            </div>
            <button
              v-if="health?.cached_example_available"
              class="button secondary small"
              :disabled="busy"
              @click="importCachedExample"
            >
              接入本机缓存示例
            </button>
          </div>
          <section class="panel table-panel">
            <div class="panel-heading">
              <h2>模型注册表</h2>
              <span>{{ models.length }} MODELS</span>
            </div>
            <div v-if="!models.length" class="empty-state">
              <span>◇</span><b>尚未接入模型</b>
              <p>可以接入已有权重，无需为了评测重新训练。</p>
              <button class="button secondary" @click="modelDialogOpen = true">
                填写模型配置
              </button>
            </div>
            <table v-else>
              <thead>
                <tr>
                  <th>模型名称</th>
                  <th>结构 / 输入</th>
                  <th>类别</th>
                  <th>状态</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="model in models" :key="model.id">
                  <td>
                    <b>{{ model.name }}</b
                    ><small class="table-detail" :title="model.weights_path">{{
                      model.weights_path
                    }}</small
                    ><small v-if="model.error" class="field-error">{{
                      model.error
                    }}</small>
                  </td>
                  <td>
                    {{ model.architecture
                    }}<small class="table-detail"
                      >RGB · {{ model.input_size }} ×
                      {{ model.input_size }}</small
                    >
                  </td>
                  <td>{{ model.num_classes }}</td>
                  <td>
                    <span class="badge" :class="model.status">{{
                      statusName(model.status)
                    }}</span>
                  </td>
                  <td>
                    <button
                      class="text-button"
                      :disabled="busy || !health?.compute_ready"
                      @click="validateModel(model)"
                    >
                      {{ busy ? "处理中…" : "验证模型" }}
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </section>
        </template>

        <DatasetPanel v-if="page === 'datasets'" />
        <DefensePanel
          v-if="page === 'defenses'"
          :models="models"
          :health="health"
        />
        <DefenseEvaluationPanel
          v-if="page === 'defense-evaluation'"
          :health="health"
        />
        <TrainingPanel
          v-if="page === 'training'"
          :health="health"
          :models="models"
        />
        <GuardPanel v-if="page === 'guard'" :health="health" />
        <ResourcePanel v-if="page === 'resources'" :health="health" />

        <BatchPanel v-if="page === 'batch'" :models="models" :health="health" />
        <SingleAttackPanel
          v-if="page === 'attack'"
          :health="health"
          :valid-models="validModels"
          :form="experimentForm"
          :catalog="attackCatalog"
          :image-preview="imagePreview"
          :busy="busy"
          @submit="startExperiment"
          @choose-image="chooseImage"
        />

        <template v-if="page === 'experiments'">
          <div class="page-heading">
            <div>
              <span class="eyebrow">EXPERIMENTS</span>
              <h1>实验记录与结果</h1>
              <p>查看实际执行状态，保留参数、模型版本和结果。</p>
            </div>
            <button class="button" @click="navigate('attack')">
              创建实验 <span>＋</span>
            </button>
          </div>
          <section class="panel table-panel">
            <div v-if="!experiments.length" class="empty-state">
              <span>▤</span><b>这里将保存你的实验</b>
              <p>提交实验后，在这里查看进度与结果。</p>
            </div>
            <table v-else>
              <thead>
                <tr>
                  <th>模型 / 实验</th>
                  <th>攻击</th>
                  <th>预算</th>
                  <th>状态与进度</th>
                  <th>时间</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="task in experiments"
                  :key="task.id"
                  class="clickable"
                  :class="{ highlighted: selectedExperimentId === task.id }"
                  @click="openExperiment(task)"
                >
                  <td>
                    <b>{{ task.model_name }}</b
                    ><small class="table-detail">{{
                      task.id.slice(0, 8)
                    }}</small>
                  </td>
                  <td>
                    {{ task.algorithm.toUpperCase()
                    }}<small
                      class="table-detail"
                      v-if="task.algorithm === 'pgd'"
                      >{{ task.steps }} steps</small
                    >
                  </td>
                  <td>{{ budgetLabel(task) }}</td>
                  <td>
                    <span class="badge" :class="task.status">{{
                      statusName(task.status)
                    }}</span
                    ><progress
                      v-if="['running', 'queued'].includes(task.status)"
                      :value="task.progress"
                      max="100"
                    ></progress>
                  </td>
                  <td>{{ formatDate(task.created_at) }}</td>
                </tr>
              </tbody>
            </table>
          </section>
          <section v-if="selectedExperiment" class="panel result-panel">
            <div class="panel-heading">
              <div>
                <h2>
                  实验详情
                  <span class="mono">{{
                    selectedExperiment.id.slice(0, 8)
                  }}</span>
                </h2>
                <small>{{ selectedExperiment.model_name }}</small>
              </div>
              <div class="inline-actions">
                <button
                  v-if="
                    ['queued', 'running', 'cancelling'].includes(
                      selectedExperiment.status,
                    )
                  "
                  class="button secondary small"
                  @click="cancelExperiment(selectedExperiment)"
                >
                  取消任务</button
                ><a
                  class="button secondary small"
                  :href="`/api/experiments/${selectedExperiment.id}/report`"
                  >导出 JSON</a
                >
              </div>
            </div>
            <div v-if="selectedExperiment.error" class="alert error">
              {{ selectedExperiment.error }}
            </div>
            <div v-if="!result" class="subtle-note">
              {{ statusName(selectedExperiment.status) }} ·
              {{ selectedExperiment.progress }}%<span
                v-if="selectedExperiment.step"
              >
                · 当前第 {{ selectedExperiment.step }} 步</span
              >
            </div>
            <template v-else
              ><div class="image-comparison">
                <figure>
                  <img :src="imageUrl('original')" alt="模型实际输入图像" />
                  <figcaption>
                    模型输入 <small>按配置缩放后的 RGB 图像</small>
                  </figcaption>
                </figure>
                <figure>
                  <img :src="imageUrl('adversarial')" alt="生成的对抗图像" />
                  <figcaption>
                    攻击输出 <small>PNG 预览，浮点数据另存</small>
                  </figcaption>
                </figure>
                <figure>
                  <img :src="imageUrl('perturbation')" alt="放大显示的扰动" />
                  <figcaption>
                    扰动 ×{{ result.perturbation_display_gain }}
                    <small>中性灰表示没有变化</small>
                  </figcaption>
                </figure>
              </div>
              <div class="result-metrics">
                <div>
                  <span>预测是否变化</span
                  ><b>{{ result.prediction_changed ? "已翻转" : "未翻转" }}</b>
                </div>
                <div>
                  <span>实际 L∞ 扰动</span
                  ><b>{{ (result.linf * 255).toFixed(3) }} / 255</b>
                </div>
                <div>
                  <span>计算耗时</span><b>{{ result.seconds }} s</b>
                </div>
                <div>
                  <span>基于真值的攻击成功</span
                  ><b>{{
                    result.attack_success === null
                      ? result.true_label === null
                        ? "未提供真值"
                        : "原预测已错误"
                      : result.attack_success
                        ? "成功"
                        : "未成功"
                  }}</b>
                </div>
              </div>
              <div class="two-columns">
                <div>
                  <h3>预测分布对比</h3>
                  <div
                    v-for="(predictions, index) in [
                      result.clean_top5,
                      result.adversarial_top5,
                    ]"
                    :key="index"
                    class="prediction-group"
                  >
                    <small>{{ index ? "攻击后" : "攻击前" }}</small>
                    <div
                      v-for="prediction in predictions.slice(0, 3)"
                      :key="prediction.index"
                      class="prediction-row"
                    >
                      <span :title="prediction.label">{{
                        prediction.label
                      }}</span>
                      <div>
                        <i
                          :style="{ width: percentage(prediction.probability) }"
                        ></i>
                      </div>
                      <b>{{ percentage(prediction.probability) }}</b>
                    </div>
                  </div>
                </div>
                <div>
                  <h3>攻击目标类别概率</h3>
                  <p v-if="!result.trace?.length" class="field-help">
                    暂无迭代曲线。
                  </p>
                  <svg
                    v-else
                    viewBox="0 0 400 200"
                    role="img"
                    aria-label="目标类别概率随攻击迭代变化"
                  >
                    <line x1="35" y1="25" x2="35" y2="165" stroke="#dce4df" />
                    <line x1="35" y1="165" x2="365" y2="165" stroke="#dce4df" />
                    <line x1="35" y1="95" x2="365" y2="95" stroke="#edf1ee" />
                    <text x="2" y="30">1.0</text>
                    <text x="2" y="170">0.0</text>
                    <text x="160" y="194">迭代步数 →</text>
                    <polyline
                      :points="tracePoints"
                      fill="none"
                      stroke="#278b67"
                      stroke-width="2.5"
                    />
                  </svg>
                </div>
              </div>
              <div class="inline-actions tensor-links">
                <a
                  :href="`/api/experiments/${selectedExperiment.id}/tensors/adversarial`"
                  >下载攻击图像 float32.npy ↗</a
                ><a
                  :href="`/api/experiments/${selectedExperiment.id}/tensors/perturbation`"
                  >下载扰动 float32.npy ↗</a
                >
              </div></template
            >
          </section>
        </template>

        <template v-if="page === 'guide'">
          <div class="page-heading">
            <div>
              <span class="eyebrow">DOCUMENTATION</span>
              <h1>使用指南</h1>
              <p>从模型接入到攻击评测、训练与实时防护。</p>
            </div>
            <a class="button secondary" href="/docs">API 文档</a>
          </div>
          <div class="two-columns">
            <section class="panel prose">
              <h2>操作流程</h2>
              <ol>
                <li>
                  在「模型接入」登记结构、权重和预处理参数，完成模型验证。
                </li>
                <li>在「数据集中心」登记训练、验证或测试数据路径。</li>
                <li>
                  在「攻击实验室」上传图片，或在「批量安全评测」选择数据集、模型和攻击。
                </li>
                <li>
                  在「防御与检测」配置输入处理、模型集成和检测器，使用验证集校准阈值。
                </li>
                <li>
                  在「攻防对比评测」比较防御配置，在「训练中心」训练或蒸馏模型。
                </li>
                <li>
                  在「实时推理防护」创建会话，通过图片请求查看预测与告警。
                </li>
              </ol>
              <h3>命令行训练</h3>
              <p>
                先安装与你的显卡和驱动匹配的
                PyTorch，再安装项目依赖，设置数据目录后运行训练命令。
              </p>
              <pre>
python -m backend.train --config configs/cifar10-pgd.json --check-only

python -m backend.train --config configs/cifar10-pgd.json

python -m backend.train --config configs/cifar10-pgd.json --resume data/training/cifar10-pgd/last.pt</pre>
              <p>
                支持
                ERM、PGD-AT、TRADES、AT-BSL、TAET、温度教师与学生蒸馏。训练中心提供任务、日志与曲线；ImageFolder
                支持独立验证目录。通过 data_dir 设置数据路径，通过 download
                下载数据。
              </p>
              <h3>模型导出与评测</h3>
              <p>
                训练生成 best_weights.pt 和
                model_registration.json。后者包含接入工作台需要的结构与预处理信息。
              </p>
              <pre>
python -m backend.evaluate_suite --config configs/evaluation-cifar10.json --output-dir data/evaluation/full-suite</pre>
            </section>
            <section class="panel prose">
              <h2>自定义模型接入</h2>
              <p>
                参考 examples/custom_model.py，提供返回 torch.nn.Module 的
                build_model()。工作台中填写加载文件的绝对路径、state_dict
                权重路径、类别数和预处理。
              </p>
              <p>
                输入为 3 通道 RGB，模型输出 [batch, num_classes] logits。
                填写后端所在机器上的模型文件路径。
              </p>
              <h3>结果与报告</h3>
              <p>
                单图实验可下载原图、对抗图和扰动张量。批量与攻防评测提供
                JSON、CSV 和 HTML 报告；训练任务提供权重、日志与曲线。
              </p>
            </section>
          </div>
        </template>
      </main>
      <footer class="work-footer">
        <span>Aegis Lab · 模型安全实验室</span>
      </footer>
    </section>
  </div>

  <div
    v-if="modelDialogOpen"
    class="modal-backdrop"
    @click.self="modelDialogOpen = false"
  >
    <form class="modal" @submit.prevent="registerModel">
      <div class="panel-heading">
        <div>
          <span class="eyebrow">REGISTER MODEL</span>
          <h2>新增 PyTorch 模型</h2>
        </div>
        <button
          type="button"
          class="close-button"
          @click="modelDialogOpen = false"
          aria-label="关闭模型接入窗口"
        >
          ×
        </button>
      </div>
      <div class="modal-content">
        <label
          >模型名称<input
            v-model="modelForm.name"
            required
            placeholder="例如：我的 CIFAR-10 分类模型"
        /></label>
        <div class="form-grid">
          <label
            >网络结构<select
              v-model="modelForm.architecture"
              @change="chooseArchitecture"
            >
              <option value="resnet18">ResNet-18</option>
              <option value="resnet34">ResNet-34</option>
              <option value="mobilenet_v2">MobileNet V2</option>
              <option value="cifar_resnet34">CIFAR ResNet-34</option>
              <option value="densenet121">DenseNet-121</option>
              <option value="efficientnet_b0">EfficientNet-B0</option>
              <option value="wide_resnet50_2">Wide ResNet-50-2</option>
              <option value="cifar_resnet18">CIFAR ResNet-18</option>
              <option value="custom">自定义加载文件</option>
            </select></label
          ><label
            >类别数量<input
              v-model.number="modelForm.num_classes"
              type="number"
              min="2"
              max="10000"
              required
          /></label>
        </div>
        <label
          >权重绝对路径<input
            v-model="modelForm.weights_path"
            required
            placeholder="/home/user/models/model.pt"
          /><small class="field-help"
            >后端所在机器上的 state_dict 权重文件。</small
          ></label
        ><template v-if="modelForm.architecture === 'custom'"
          ><label
            >Python 加载文件<input
              v-model="modelForm.loader_path"
              required
              placeholder="/home/user/my_model.py" /></label
          ><label
            >构造函数名<input v-model="modelForm.factory" required /></label
        ></template>
        <div class="form-grid">
          <label
            >输入尺寸<input
              v-model.number="modelForm.input_size"
              type="number"
              min="16"
              max="1024"
              required
          /></label>
          <div class="read-only-field">
            输入格式<b>RGB · [0, 1] · 正方形缩放</b>
          </div>
        </div>
        <div class="form-grid">
          <label>归一化 mean<input v-model="modelForm.mean" required /></label
          ><label>归一化 std<input v-model="modelForm.std" required /></label>
        </div>
        <label
          >类别名称 <span class="optional">可选，每行一个</span
          ><textarea
            v-model="modelForm.categories"
            rows="3"
            placeholder="留空显示类别编号；填写时数量与顺序必须匹配模型"
          ></textarea>
        </label>
      </div>
      <div class="modal-footer">
        <button
          class="button secondary"
          type="button"
          @click="modelDialogOpen = false"
        >
          取消</button
        ><button class="button" :disabled="busy" type="submit">
          {{ busy ? "保存中…" : "保存模型配置" }}
        </button>
      </div>
    </form>
  </div>
</template>
