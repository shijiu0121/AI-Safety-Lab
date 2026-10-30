import { computed, onMounted, onUnmounted, ref } from "vue";

import { apiRequest } from "./http";

export function useLab() {
  // 页面导航、服务状态和实验数据都从这里进入，没有前端编造的统计数据。
  const page = ref(location.hash.replace(/^#\/workbench\/?/, "") || "overview");
  const health = ref(null);
  const workspaceSummary = ref(null);
  const models = ref([]);
  const attackCatalog = ref([]);
  const experiments = ref([]);
  const error = ref("");
  const notice = ref("");
  const busy = ref(false);
  const selectedExperimentId = ref(null);
  const modelDialogOpen = ref(false);
  const uploadedImage = ref(null);
  const imagePreview = ref("");
  let refreshTimer;

  const navigation = [
    { id: "overview", icon: "◫", label: "工作台概览" },
    { id: "models", icon: "◇", label: "模型接入" },
    { id: "datasets", icon: "▦", label: "数据集中心" },
    { id: "attack", icon: "⌁", label: "攻击实验室" },
    { id: "batch", icon: "▧", label: "批量安全评测" },
    { id: "defenses", icon: "◈", label: "防御与检测" },
    { id: "defense-evaluation", icon: "⇄", label: "攻防对比评测" },
    { id: "training", icon: "△", label: "训练中心" },
    { id: "guard", icon: "◉", label: "实时推理防护" },
    { id: "resources", icon: "▱", label: "资源管理" },
    { id: "experiments", icon: "▤", label: "实验记录" },
    { id: "guide", icon: "≡", label: "使用指南" },
  ];
  const titles = Object.fromEntries(
    navigation.map((item) => [item.id, item.label]),
  );
  const validModels = computed(() =>
    models.value.filter((model) => model.status === "validated"),
  );
  const runningCount = computed(() => workspaceSummary.value?.running || 0);
  const completedCount = computed(() => workspaceSummary.value?.completed || 0);
  const selectedExperiment = computed(() =>
    experiments.value.find((task) => task.id === selectedExperimentId.value),
  );
  const result = computed(() => selectedExperiment.value?.result);
  const experimentForm = ref({
    model_id: "",
    algorithm: "pgd20",
    epsilon_l2: 0.5,
    restarts: 1,
    cw_steps: 100,
    cw_confidence: 0,
    deepfool_steps: 50,
    deepfool_candidates: 10,
    epsilon_255: 8,
    steps: 20,
    step_size_255: 1,
    seed: 0,
    true_label: "",
  });
  const modelForm = ref({
    name: "",
    architecture: "resnet18",
    weights_path: "",
    loader_path: "",
    factory: "build_model",
    num_classes: 1000,
    input_size: 224,
    mean: "0.485, 0.456, 0.406",
    std: "0.229, 0.224, 0.225",
    categories: "",
  });

  function navigate(destination) {
    location.hash = destination === "home" ? "/home" : "/workbench/" + destination;
    window.scrollTo(0, 0);
  }
  function onHashChange() {
    page.value = location.hash.replace(/^#\/workbench\/?/, "") || "overview";
  }
  function formatDate(value) {
    return value
      ? new Date(value).toLocaleString("zh-CN", { hour12: false })
      : "—";
  }
  function percentage(value) {
    return (value * 100).toFixed(2) + "%";
  }
  const statusNames = {
    registered: "待验证",
    validated: "已验证",
    invalid: "验证失败",
    queued: "排队中",
    running: "运行中",
    cancelling: "取消中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
    interrupted: "服务中断",
  };
  function statusName(value) {
    return statusNames[value] || value;
  }

  const api = apiRequest;
  async function refresh() {
    try {
      const responses = await Promise.all([
        api("/health"),
        api("/models"),
        api("/experiments"),
        api("/attacks"),
        api("/workspace-summary"),
      ]);
      [
        health.value,
        models.value,
        experiments.value,
        attackCatalog.value,
        workspaceSummary.value,
      ] = responses;
      if (!experimentForm.value.model_id && validModels.value.length)
        experimentForm.value.model_id = validModels.value[0].id;
    } catch (problem) {
      health.value = null;
      error.value = "无法连接计算服务：" + problem.message;
    }
  }
  async function perform(action) {
    busy.value = true;
    error.value = "";
    notice.value = "";
    try {
      await action();
      await refresh();
    } catch (problem) {
      error.value = problem.message;
    } finally {
      busy.value = false;
    }
  }
  async function registerModel() {
    await perform(async () => {
      const form = modelForm.value;
      const payload = {
        ...form,
        mean: form.mean.split(",").map(Number),
        std: form.std.split(",").map(Number),
        categories: form.categories.trim()
          ? form.categories
              .trim()
              .split("\n")
              .map((label) => label.trim())
          : [],
      };
      await api("/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      modelDialogOpen.value = false;
      notice.value = "模型配置已登记。下一步执行兼容性验证。";
    });
  }
  async function importCachedExample() {
    await perform(async () => {
      await api("/models/import-cached-example", { method: "POST" });
      notice.value = "已接入本机缓存模型。";
    });
  }
  async function validateModel(model) {
    await perform(async () => {
      await api(`/models/${model.id}/validate`, { method: "POST" });
      notice.value = "模型推理、输出形状和输入梯度验证通过。";
    });
  }
  function chooseArchitecture() {
    if (
      ["cifar_resnet18", "cifar_resnet34"].includes(
        modelForm.value.architecture,
      )
    )
      Object.assign(modelForm.value, {
        num_classes: 10,
        input_size: 32,
        mean: "0.4914, 0.4822, 0.4465",
        std: "0.2470, 0.2435, 0.2616",
      });
  }
  function chooseImage(event) {
    uploadedImage.value = event.target.files[0] || null;
    if (imagePreview.value) URL.revokeObjectURL(imagePreview.value);
    imagePreview.value = uploadedImage.value
      ? URL.createObjectURL(uploadedImage.value)
      : "";
  }
  async function startExperiment() {
    await perform(async () => {
      if (!uploadedImage.value) throw new Error("请选择待评测图片。");
      const config = {
        ...experimentForm.value,
        true_label:
          experimentForm.value.true_label === ""
            ? null
            : Number(experimentForm.value.true_label),
      };
      const form = new FormData();
      form.append("config", JSON.stringify(config));
      form.append("image", uploadedImage.value);
      const task = await api("/experiments", { method: "POST", body: form });
      selectedExperimentId.value = task.id;
      navigate("experiments");
      notice.value = "实验已提交，运行进度来自后端计算过程。";
    });
  }
  function openExperiment(task) {
    selectedExperimentId.value = task.id;
    navigate("experiments");
  }
  async function cancelExperiment(task) {
    await perform(() =>
      api(`/experiments/${task.id}/cancel`, { method: "POST" }),
    );
  }
  function imageUrl(kind) {
    return `/api/experiments/${selectedExperimentId.value}/images/${kind}`;
  }
  const tracePoints = computed(() => {
    const trace = result.value?.trace || selectedExperiment.value?.trace || [];
    return trace
      .map(
        (point, index) =>
          `${35 + (index / Math.max(trace.length - 1, 1)) * 330},${165 - point.target_probability * 140}`,
      )
      .join(" ");
  });
  onMounted(() => {
    window.addEventListener("hashchange", onHashChange);
    refresh();
    refreshTimer = setInterval(() => {
      if (page.value !== "home" && !busy.value) refresh();
    }, 2000);
  });
  onUnmounted(() => {
    clearInterval(refreshTimer);
    window.removeEventListener("hashchange", onHashChange);
    if (imagePreview.value) URL.revokeObjectURL(imagePreview.value);
  });
  function attackName(task) {
    return (
      task.attack_protocol?.name ||
      attackCatalog.value.find((item) => item.id === task.algorithm)?.name ||
      task.algorithm.toUpperCase()
    );
  }
  function budgetLabel(task) {
    return task.attack_protocol?.norm === "L2"
      ? "L2 " + task.epsilon_l2
      : task.epsilon_255 + "/255";
  }
  return {
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
  };
}
