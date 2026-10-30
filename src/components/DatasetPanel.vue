<script setup>
import { onMounted, ref } from "vue";
import { apiRequest } from "../http";

const datasets = ref([]);
const error = ref("");
const busy = ref(false);
const showForm = ref(false);
const selected = ref(null);
const form = ref({ name: "", kind: "cifar10", root: "", split: "test" });
const statusNames = {
  registered: "待检查",
  validated: "已检查",
  invalid: "检查失败",
};

async function refresh() {
  datasets.value = await apiRequest("/datasets");
}
async function save() {
  busy.value = true;
  error.value = "";
  try {
    await apiRequest("/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form.value),
    });
    showForm.value = false;
    await refresh();
  } catch (problem) {
    error.value = problem.message;
  } finally {
    busy.value = false;
  }
}
async function inspect(dataset) {
  busy.value = true;
  error.value = "";
  try {
    selected.value = await apiRequest(`/datasets/${dataset.id}/inspect`, {
      method: "POST",
    });
  } catch (problem) {
    error.value = problem.message;
  } finally {
    busy.value = false;
    await refresh();
  }
}
onMounted(async () => {
  try {
    await refresh();
  } catch (problem) {
    error.value = problem.message;
  }
});
</script>

<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">DATASET REGISTRY</span>
      <h1>数据集中心</h1>
      <p>登记数据路径、划分与类别顺序，查看样本分布。</p>
    </div>
    <button class="button" @click="showForm = !showForm">登记数据集 ＋</button>
  </div>
  <div v-if="error" class="alert error" role="alert">{{ error }}</div>
  <form
    v-if="showForm"
    class="panel form-panel dataset-form"
    @submit.prevent="save"
  >
    <h2>登记一个评测数据集</h2>
    <div class="form-grid">
      <label
        >名称<input
          v-model="form.name"
          required
          placeholder="例如：CIFAR-10 官方测试集" /></label
      ><label
        >格式<select v-model="form.kind" @change="form.split = 'test'">
          <option value="cifar10">CIFAR-10</option>
          <option value="cifar100">CIFAR-100</option>
          <option value="imagefolder">ImageFolder 自定义目录</option>
        </select></label
      >
    </div>
    <label
      >服务器上的数据根目录<input
        v-model="form.root"
        required
        placeholder="/home/user/datasets/cifar"
      /><small class="field-help"
        >可先登记服务器路径，迁移后再检查。CIFAR 指解压数据的父目录；ImageFolder
        指类别文件夹的父目录。</small
      ></label
    >
    <label
      >划分<select v-model="form.split" :disabled="form.kind !== 'imagefolder'">
        <option value="test">测试集</option>
        <option value="validation">独立验证集</option>
      </select></label
    >
    <p class="field-help">
      自定义目录结构为
      root/类别A/图片、root/类别B/图片。类别顺序由目录名排序决定，必须与模型输出顺序一致。
    </p>
    <button class="button" type="submit" :disabled="busy">保存配置</button>
  </form>
  <section class="panel table-panel">
    <div class="panel-heading">
      <h2>数据资源</h2>
      <span>{{ datasets.length }} DATASETS</span>
    </div>
    <div v-if="!datasets.length" class="empty-state">
      <span>▦</span><b>还没有登记数据集</b>
      <p>模型与数据确认兼容后，才可以创建批量实验。</p>
    </div>
    <table v-else>
      <thead>
        <tr>
          <th>名称 / 路径</th>
          <th>格式与划分</th>
          <th>样本 / 类别</th>
          <th>状态</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="dataset in datasets" :key="dataset.id">
          <td>
            <b>{{ dataset.name }}</b
            ><small class="table-detail" :title="dataset.root">{{
              dataset.root
            }}</small
            ><small v-if="dataset.error" class="field-error">{{
              dataset.error
            }}</small>
          </td>
          <td>
            {{ dataset.kind
            }}<small class="table-detail">{{ dataset.split }}</small>
          </td>
          <td>
            {{ dataset.inspection?.sample_count ?? "—" }} /
            {{ dataset.inspection?.classes.length ?? "—" }}
          </td>
          <td>
            <span class="badge" :class="dataset.status">{{
              statusNames[dataset.status]
            }}</span>
          </td>
          <td>
            <button
              class="text-button"
              :disabled="busy"
              @click="inspect(dataset)"
            >
              检查目录</button
            ><button
              v-if="dataset.inspection"
              class="text-button"
              @click="selected = dataset"
            >
              查看类别
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </section>
  <section v-if="selected?.inspection" class="panel class-distribution">
    <div class="panel-heading">
      <h2>{{ selected.name }} · 类别分布</h2>
      <span>当前数据划分</span>
    </div>
    <div class="class-bars">
      <div v-for="(name, index) in selected.inspection.classes" :key="name">
        <span>{{ index }} · {{ name }}</span>
        <div>
          <i
            :style="{
              width:
                (selected.inspection.class_counts[index] /
                  Math.max(...selected.inspection.class_counts)) *
                  100 +
                '%',
            }"
          ></i>
        </div>
        <b>{{ selected.inspection.class_counts[index] }}</b>
      </div>
    </div>
    <p class="field-help">
      {{ selected.inspection.note }}
      当前数据划分的类别样本数量。
    </p>
  </section>
</template>
