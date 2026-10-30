// 所有 JSON 接口统一处理失败信息，文件下载仍使用普通链接。
export async function apiRequest(path, options = {}) {
  const response = await fetch("/api" + path, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail || body),
    );
  }
  return body;
}
