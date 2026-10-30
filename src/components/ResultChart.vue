<script setup>
import { computed } from "vue";
const props = defineProps({
  series: { type: Array, default: () => [] },
  xLabel: { type: String, default: "扰动预算 ε / 255" },
  yLabel: { type: String, default: "鲁棒准确率" },
  percentage: { type: Boolean, default: true },
});
const colors = [
  "#167653",
  "#8b72b6",
  "#d89642",
  "#427ba4",
  "#b65e69",
  "#7d8939",
];
const points = computed(() => props.series.flatMap((s) => s.points));
const xmax = computed(() => Math.max(1, ...points.value.map((p) => p.x)));
const ymax = computed(() =>
  props.percentage ? 1 : Math.max(1, ...points.value.map((p) => p.y)) * 1.05,
);
const px = (x) => 65 + (x / xmax.value) * 650,
  py = (y) => 265 - (y / ymax.value) * 220;
const ticks = computed(() =>
  Array.from(new Set(points.value.map((p) => p.x)))
    .sort((a, b) => a - b)
    .filter((_, i, a) => a.length < 10 || i % Math.ceil(a.length / 8) === 0),
);
const format = (y) =>
  props.percentage ? (y * 100).toFixed(0) + "%" : y.toFixed(2);
</script>
<template>
  <div class="result-chart">
    <svg
      viewBox="0 0 760 325"
      role="img"
      :aria-label="`${yLabel}与${xLabel}对比图`"
    >
      <text x="65" y="20">{{ yLabel }}</text>
      <g v-for="tick in [0, 0.25, 0.5, 0.75, 1]" :key="tick">
        <line
          x1="65"
          x2="715"
          :y1="py(tick * ymax)"
          :y2="py(tick * ymax)"
          stroke="#e1e8e3"
        />
        <text x="53" :y="py(tick * ymax) + 4" text-anchor="end">
          {{ format(tick * ymax) }}
        </text>
      </g>
      <text
        v-for="tick in ticks"
        :key="tick"
        :x="px(tick)"
        y="288"
        text-anchor="middle"
      >
        {{ tick }}
      </text>
      <text x="390" y="316" text-anchor="middle">{{ xLabel }}</text>
      <g v-for="(line, i) in series" :key="line.name">
        <polyline
          fill="none"
          :stroke="colors[i % colors.length]"
          stroke-width="2.5"
          :points="line.points.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')"
        />
        <circle
          v-for="(point, j) in line.points"
          :key="j"
          :cx="px(point.x)"
          :cy="py(point.y)"
          r="3"
          :fill="colors[i % colors.length]"
        >
          <title>{{ line.name }} · {{ point.x }}：{{ format(point.y) }}</title>
        </circle>
      </g>
    </svg>
    <div class="chart-legend">
      <span v-for="(line, i) in series" :key="line.name"
        ><i :style="{ background: colors[i % colors.length] }"></i
        >{{ line.name }}</span
      >
    </div>
  </div>
</template>
<style scoped>
svg {
  width: 100%;
  display: block;
}
text {
  font-size: 12px;
  fill: #718279;
  font-family: inherit;
}
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 20px;
  font-size: 12px;
  padding: 0 28px 18px;
}
.chart-legend span {
  display: flex;
  align-items: center;
  gap: 7px;
}
.chart-legend i {
  width: 14px;
  height: 3px;
  display: inline-block;
}
</style>
