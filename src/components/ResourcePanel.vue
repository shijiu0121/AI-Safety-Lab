<script setup>
import { onMounted, ref } from "vue";
import { apiRequest } from "../http";
const props = defineProps({ health: Object });
const kind = ref("models"),
  items = ref([]),
  archived = ref([]),
  editing = ref(""),
  text = ref(""),
  error = ref(""),
  notice = ref(""),
  busy = ref(false);
const modelFields = [
  "name",
  "architecture",
  "weights_path",
  "loader_path",
  "factory",
  "num_classes",
  "input_size",
  "mean",
  "std",
  "categories",
];
const datasetFields = ["name", "kind", "root", "split"];
async function refresh() {
  [items.value, archived.value] = await Promise.all([
    apiRequest("/" + kind.value),
    apiRequest("/resources/archived"),
  ]);
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
function select(item) {
  editing.value = item.id;
  const fields = kind.value === "models" ? modelFields : datasetFields;
  text.value = JSON.stringify(
    Object.fromEntries(
      fields
        .filter((field) => field in item)
        .map((field) => [field, item[field]]),
    ),
    null,
    2,
  );
  notice.value = "资源配置已载入，修改路径后请重新检查。";
}
function newResource() {
  editing.value = "";
  text.value = JSON.stringify(
    kind.value === "models"
      ? {
          name: "",
          architecture: "cifar_resnet18",
          weights_path: "/path/to/weights.pt",
          loader_path: "",
          factory: "build_model",
          num_classes: 10,
          input_size: 32,
          mean: [0.4914, 0.4822, 0.4465],
          std: [0.247, 0.2435, 0.2616],
          categories: [],
        }
      : {
          name: "",
          kind: "imagefolder",
          root: "/path/to/images",
          split: "test",
        },
    null,
    2,
  );
}
async function save() {
  await action(async () => {
    const resource = JSON.parse(text.value);
    await apiRequest(
      editing.value ? `/${kind.value}/${editing.value}` : "/resources/import",
      {
        method: editing.value ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          editing.value ? resource : { kind: kind.value, resource },
        ),
      },
    );
    notice.value = "资源已保存，验证状态已重置。";
    editing.value = "";
  });
}
async function readFile(event) {
  error.value = "";
  try {
    text.value = await event.target.files[0].text();
    editing.value = "";
  } catch (problem) {
    error.value = problem.message;
  }
}
async function switchKind() {
  editing.value = "";
  text.value = "";
  await action(refresh);
}
onMounted(() => action(refresh));
</script>
<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">RESOURCE MANAGER</span>
      <h1>资源管理与迁移</h1>
      <p>导入训练产物、修改文件路径和模型预处理参数。</p>
    </div>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <div v-if="notice" class="alert success">{{ notice }}</div>
  <section class="panel form-panel">
    <div class="form-grid">
      <label
        >资源类型<select v-model="kind" @change="switchKind">
          <option value="models">模型</option>
          <option value="datasets">数据集</option>
        </select></label
      ><label
        >读取本地 JSON 文件<input
          type="file"
          accept="application/json,.json"
          @change="readFile"
      /></label>
    </div>
    <div class="inline-actions wrap">
      <button class="button secondary small" @click="newResource">
        填写新资源模板</button
      ><a href="#/workbench/defenses">防御配置在工作室编辑 →</a>
    </div>
    <div class="table-scroll module-section">
      <table>
        <thead>
          <tr>
            <th>资源</th>
            <th>当前路径</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in items" :key="item.id">
            <td>{{ item.name }}</td>
            <td>
              <small class="path-text">{{
                item.weights_path || item.root
              }}</small>
            </td>
            <td>{{ item.status }}</td>
            <td>
              <div class="inline-actions wrap">
                <button class="text-button" @click="select(item)">编辑</button
                ><a :href="`/api/resources/${kind}/${item.id}/export`">JSON</a
                ><button
                  class="text-button"
                  :disabled="
                    busy || (kind === 'models' && !health?.compute_ready)
                  "
                  @click="
                    action(() =>
                      apiRequest(
                        kind === 'models'
                          ? `/models/${item.id}/validate`
                          : `/datasets/${item.id}/inspect`,
                        { method: 'POST' },
                      ),
                    )
                  "
                >
                  {{ kind === "models" ? "验证" : "检查" }}</button
                ><button
                  class="text-button"
                  :disabled="busy"
                  @click="
                    action(() =>
                      apiRequest(`/resources/${kind}/${item.id}/archive`, {
                        method: 'POST',
                      }),
                    )
                  "
                >
                  归档
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-if="!items.length" class="empty-copy">
      尚无资源，可以导入 model_registration.json 或手动填写。
    </p>
    <form v-if="text" class="module-section" @submit.prevent="save">
      <h3>{{ editing ? "编辑现有资源" : "导入为新资源" }}</h3>
      <label
        >完整资源 JSON<textarea
          v-model="text"
          class="config-editor"
          rows="18"
          required
          spellcheck="false"
        ></textarea>
      </label>
      <p class="field-help">
        路径属于后端运行机器。系统元数据在导入时会剥离，未知配置字段会报错。自定义
        Python 加载文件需由你信任。
      </p>
      <button class="button small" :disabled="busy">校验并保存</button>
    </form>
  </section>
  <section class="panel module-section">
    <h2>已归档资源</h2>
    <p class="field-help">归档资源可以恢复。归档前请先停止相关任务或会话。</p>
    <div v-for="item in archived" :key="item.id" class="job-card">
      <b>{{ item.name }}</b
      ><span>{{ item.kind }}</span
      ><button
        class="text-button"
        @click="
          action(() =>
            apiRequest(`/resources/${item.kind}/${item.id}/restore`, {
              method: 'POST',
            }),
          )
        "
      >
        恢复
      </button>
    </div>
    <p v-if="!archived.length" class="empty-copy">暂无归档。</p>
  </section>
</template>
