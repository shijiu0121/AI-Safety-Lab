<script setup>
import { ref, computed, onUnmounted, defineAsyncComponent } from "vue";
import PublicSite from "./PublicSite.vue";
const Workbench = defineAsyncComponent(() => import("./Workbench.vue"));
const hash = ref(location.hash);
const update = () => {
  hash.value = location.hash;
  window.scrollTo(0, 0);
};
window.addEventListener("hashchange", update);
onUnmounted(() => window.removeEventListener("hashchange", update));
const workbench = computed(() => hash.value.startsWith("#/workbench"));
</script>
<template><Workbench v-if="workbench" /><PublicSite v-else /></template>
