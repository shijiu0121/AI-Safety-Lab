<script setup>
import { onMounted, onUnmounted, ref } from "vue";
import { apiRequest } from "../http";
const props = defineProps({ health: Object });
const profiles = ref([]),
  sessions = ref([]),
  selected = ref(""),
  stats = ref(null),
  latest = ref(null),
  file = ref(null),
  error = ref(""),
  busy = ref(false);
const form = ref({
  name: "",
  defense_id: "",
  policy: "monitor",
  save_images: false,
});
let timer;
async function refresh() {
  [profiles.value, sessions.value] = await Promise.all([
    apiRequest("/defenses"),
    apiRequest("/guard/sessions"),
  ]);
  stats.value = await apiRequest(
    `/guard/events?session_id=${encodeURIComponent(selected.value)}`,
  );
}
async function action(callback) {
  busy.value = true;
  error.value = "";
  try {
    await callback();
    await refresh();
  } catch (problem) {
    error.value = problem.message;
  } finally {
    busy.value = false;
  }
}
async function create() {
  await action(async () => {
    const session = await apiRequest("/guard/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form.value),
    });
    selected.value = session.id;
  });
}
async function predict() {
  await action(async () => {
    if (!file.value) throw new Error("先选择一张图片。");
    const payload = new FormData();
    payload.append("image", file.value);
    latest.value = await apiRequest(
      `/guard/sessions/${selected.value}/predict`,
      { method: "POST", body: payload },
    );
  });
}
const seconds = (value) => (value == null ? "—" : value.toFixed(4) + " s");
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
      <span class="eyebrow">LIVE INFERENCE GUARD</span>
      <h1>实时推理防护</h1>
      <p>接收图片请求，执行处理、预测与检测，记录放行或拦截结果。</p>
    </div>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <form class="panel form-panel" @submit.prevent="create">
    <h2>创建防护会话</h2>
    <div class="form-grid">
      <label
        >会话名称<input
          v-model="form.name"
          required
          placeholder="例如：模型推理防护演示" /></label
      ><label
        >防御流程<select v-model="form.defense_id" required>
          <option disabled value="">选择防御配置</option>
          <option
            v-for="profile in profiles"
            :key="profile.id"
            :value="profile.id"
          >
            {{ profile.name }}
          </option>
        </select></label
      ><label
        >决策策略<select v-model="form.policy">
          <option value="monitor">监测：告警但继续返回预测</option>
          <option value="reject">拦截：命中规则后拒绝预测</option>
        </select></label
      >
    </div>
    <div class="checkbox-grid">
      <label
        ><input
          v-model="form.save_images"
          type="checkbox"
        />在本实例保存请求原图与处理预览（默认不保存）</label
      >
    </div>
    <button class="button" :disabled="busy || !profiles.length">
      创建会话
    </button>
  </form>
  <section class="panel form-panel module-section">
    <div class="form-grid">
      <label
        >查看会话<select v-model="selected" @change="refresh">
          <option value="">所有会话的历史统计</option>
          <option
            v-for="session in sessions"
            :key="session.id"
            :value="session.id"
          >
            {{ session.name }} · {{ session.active ? "运行" : "停止" }} ·
            {{ session.policy }}
          </option>
        </select></label
      >
    </div>
    <div v-if="selected" class="inline-actions wrap">
      <code>POST /api/guard/sessions/{{ selected }}/predict</code
      ><button
        v-if="sessions.find((item) => item.id === selected)?.active"
        class="text-button"
        @click="
          action(() =>
            apiRequest(`/guard/sessions/${selected}/stop`, { method: 'POST' }),
          )
        "
      >
        停止此会话
      </button>
    </div>
    <form v-if="selected" class="module-section" @submit.prevent="predict">
      <h3>发送测试图片</h3>
      <label
        >图片<input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          @change="file = $event.target.files[0]" /></label
      ><button
        class="button small"
        :disabled="
          busy ||
          !health?.compute_ready ||
          !sessions.find((item) => item.id === selected)?.active
        "
      >
        执行防护推理 →
      </button>
      <p class="field-help">与训练、评测共用任务队列。</p>
    </form>
    <div
      v-if="latest"
      class="guard-verdict"
      :class="latest.flagged ? 'flagged' : 'passed'"
    >
      <b
        >{{ latest.decision === "reject" ? "已拒绝" : "已放行"
        }}{{ latest.flagged ? " · 命中规则" : "" }}</b
      >
      <p>
        预测 {{ latest.prediction ?? "未返回" }} · 检测分数
        {{ latest.score.toFixed(5) }} · 耗时 {{ seconds(latest.seconds) }}
      </p>
      <p>{{ latest.reasons.join(" / ") || "未命中配置规则" }}</p>
      <div v-if="latest.images_saved" class="guard-images">
        <img
          :src="`/api/guard/events/${latest.id}/images/original`"
          alt="本次原图"
        /><img
          :src="`/api/guard/events/${latest.id}/images/processed`"
          alt="本次处理图预览"
        />
      </div>
    </div>
  </section>
  <div class="stat-grid guard-stats module-section">
    <article class="stat-card">
      <span>实际请求</span><b>{{ stats?.total ?? 0 }}</b>
    </article>
    <article class="stat-card">
      <span>规则告警</span><b>{{ stats?.flagged ?? 0 }}</b>
    </article>
    <article class="stat-card">
      <span>拒绝 / 失败</span
      ><b>{{ stats?.rejected ?? 0 }} / {{ stats?.errors ?? 0 }}</b>
    </article>
    <article class="stat-card">
      <span>平均 / P95 耗时</span
      ><b class="small-stat"
        >{{ seconds(stats?.mean_seconds) }} /
        {{ seconds(stats?.p95_seconds) }}</b
      >
    </article>
  </div>
  <section class="panel module-section">
    <div class="panel-heading">
      <h2>防护事件</h2>
      <span>最近 {{ stats?.events.length ?? 0 }} 条</span>
    </div>
    <p class="field-help">请求耗时包含模型加载、输入处理、检测和推理。</p>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>时间 / 输入</th>
            <th>决策</th>
            <th>检测分数</th>
            <th>原因</th>
            <th>耗时</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="event in stats?.events || []" :key="event.id">
            <td>
              {{ new Date(event.created_at).toLocaleString()
              }}<small class="table-detail">{{
                event.original_filename
              }}</small>
            </td>
            <td>
              <span
                class="badge"
                :class="event.decision === 'reject' ? 'invalid' : 'validated'"
                >{{ event.decision }} / {{ event.status }}</span
              >
            </td>
            <td>{{ event.score?.toFixed(5) ?? "—" }}</td>
            <td>
              {{ event.error || event.reasons?.join(" / ") || "未命中规则" }}
            </td>
            <td>{{ seconds(event.seconds) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-if="!stats?.events.length" class="empty-copy">暂无请求。</p>
  </section>
</template>
