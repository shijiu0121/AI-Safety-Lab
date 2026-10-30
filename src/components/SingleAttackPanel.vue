<script setup>
import { computed } from "vue";
const props = defineProps({
  health: Object,
  validModels: Array,
  form: Object,
  catalog: Array,
  imagePreview: String,
  busy: Boolean,
});
defineEmits(["submit", "choose-image"]);
const selectedAttack = computed(() =>
  props.catalog.find((item) => item.id === props.form.algorithm),
);
const selectedModel = computed(() =>
  props.validModels.find((model) => model.id === props.form.model_id),
);
const isL2 = computed(() => selectedAttack.value?.norm === "L2");
const iterative = computed(() =>
  ["pgd", "pgd20", "pgd100", "cw_linf"].includes(props.form.algorithm),
);
const compatible = computed(
  () =>
    selectedModel.value &&
    selectedAttack.value &&
    selectedModel.value.num_classes >= selectedAttack.value.min_classes &&
    selectedAttack.value.available,
);
</script>
<template>
  <div class="page-heading">
    <div>
      <span class="eyebrow">ATTACK LAB</span>
      <h1>配置一次对抗实验</h1>
      <p>
        PGD-20、PGD-100、CW-L∞、官方 AA standard 与 DeepFool。原始像素范围为
        [0,1]。
      </p>
    </div>
  </div>
  <form class="two-columns attack-layout" @submit.prevent="$emit('submit')">
    <section class="panel form-panel">
      <h2><span class="step-number">1</span>模型与输入</h2>
      <label
        >待评测模型<select v-model="form.model_id" required>
          <option value="" disabled>选择已通过验证的模型</option>
          <option
            v-for="model in validModels"
            :key="model.id"
            :value="model.id"
          >
            {{ model.name }}
          </option>
        </select></label
      >
      <p v-if="!validModels.length" class="field-help">
        暂无可用模型。<a href="#/workbench/models">先接入并验证模型 →</a>
      </p>
      <label class="upload-area"
        ><input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          @change="$emit('choose-image', $event)"
        /><img
          v-if="imagePreview"
          :src="imagePreview"
          alt="待评测原图"
        /><template v-else
          ><span>＋</span><b>选择一张图片</b
          ><small>PNG / JPEG / WEBP · 最大 12 MB</small></template
        ></label
      ><label
        >真实类别编号 <span class="optional">可选</span
        ><input
          v-model="form.true_label"
          type="number"
          min="0"
          placeholder="留空时评测原预测是否翻转"
      /></label>
    </section>
    <section class="panel form-panel">
      <h2><span class="step-number">2</span>攻击协议</h2>
      <label
        >攻击方法<select v-model="form.algorithm" aria-label="攻击方法">
          <option
            v-for="attack in catalog"
            :key="attack.id"
            :value="attack.id"
            :disabled="!attack.available"
          >
            {{ attack.name }}{{ attack.available ? "" : "（缺少依赖）" }}
          </option>
        </select></label
      >
      <label v-if="isL2"
        >L2 扰动预算<input
          v-model.number="form.epsilon_l2"
          type="number"
          min="0"
          max="20"
          step="0.05"
          required /></label
      ><template v-else
        ><label
          >扰动预算 ε <b class="right-label">{{ form.epsilon_255 }}/255</b
          ><input
            v-model.number="form.epsilon_255"
            type="range"
            min="0"
            max="32"
            step="1"
        /></label>
        <div class="range-labels">
          <span>0 / 255</span><span>L∞ 约束</span><span>32 / 255</span>
        </div></template
      >
      <div v-if="iterative" class="form-grid">
        <label
          >步长（/255）<input
            v-model.number="form.step_size_255"
            type="number"
            min="0.001"
            max="32"
            step="0.001"
            required /></label
        ><label
          >重启次数<input
            v-model.number="form.restarts"
            type="number"
            min="1"
            max="20"
            required
        /></label>
      </div>
      <label v-if="form.algorithm === 'pgd'"
        >自定义 PGD 步数<input
          v-model.number="form.steps"
          type="number"
          min="1"
          max="500"
          required
      /></label>
      <div v-if="form.algorithm === 'cw_linf'" class="form-grid">
        <label
          >CW 步数<input
            v-model.number="form.cw_steps"
            type="number"
            min="1"
            max="500"
            required /></label
        ><label
          >置信参数 κ<input
            v-model.number="form.cw_confidence"
            type="number"
            min="0"
            step="0.1"
            required
        /></label>
      </div>
      <div v-if="form.algorithm.startsWith('deepfool')" class="form-grid">
        <label
          >最大步数<input
            v-model.number="form.deepfool_steps"
            type="number"
            min="1"
            max="500"
            required /></label
        ><label
          >候选类别数<input
            v-model.number="form.deepfool_candidates"
            type="number"
            min="2"
            max="1000"
            required
        /></label>
      </div>
      <label
        >随机种子<input
          v-model.number="form.seed"
          type="number"
          min="0"
          required
      /></label>
      <div class="subtle-note">
        <template v-if="form.algorithm === 'autoattack'"
          >AA standard：APGD-CE、APGD-T、FAB-T、Square。</template
        ><template v-else-if="form.algorithm === 'cw_linf'"
          >CW-L∞ 使用 margin 损失，按设置的预算和步数迭代。</template
        ><template v-else-if="form.algorithm.startsWith('deepfool')"
          >DeepFool 使用 Foolbox，支持 L∞ 与 L2 扰动预算。</template
        ><template v-else
          >PGD-20 和 PGD-100 分别固定为 20 与 100
          步；在模型归一化之前约束扰动，保留迭代中已找到的成功样本。</template
        >
      </div>
      <p
        v-if="selectedModel && selectedAttack && !compatible"
        class="field-error"
      >
        当前模型或依赖与此协议不兼容，请检查类别数和攻击依赖。
      </p>
      <button
        class="button full-width"
        type="submit"
        :disabled="busy || !health?.compute_ready || !compatible"
      >
        {{ busy ? "提交中…" : "提交 GPU 攻击任务" }} →
      </button>
    </section>
  </form>
</template>
