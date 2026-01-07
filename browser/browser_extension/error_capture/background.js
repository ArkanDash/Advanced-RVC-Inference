// 存储请求信息的Map，以requestId为key
const requestMap = new Map();

// Supabase请求URL匹配规则
const SUPABASE_PATTERNS = [
  "*://*.supabase.co/rest/*",  // REST API
  "*://*.supabase.co/functions/*",  // Edge Functions
  "*://*.supabase.co/auth/*",    // Auth API
  "*://*.supabase.co/storage/*"  // Storage API
];

// 从URL中提取API类型和路径
function extractApiInfo(url) {
  try {
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/');
    const apiType = pathParts[1]; // rest, functions, auth
    const apiPath = pathParts.slice(3).join('/'); // 去掉版本号的路径
    return {
      projectId: urlObj.host.split('.')[0],
      apiType,
      apiPath,
      query: urlObj.search
    };
  } catch (e) {
    return {
      projectId: 'unknown',
      apiType: 'unknown',
      apiPath: url,
      query: ''
    };
  }
}

// 监听网络请求的开始
chrome.webRequest.onBeforeRequest.addListener(
  (details) => {
    const apiInfo = extractApiInfo(details.url);
    console.log(`[Matrix] 捕获到 Supabase ${apiInfo.apiType} 请求:`, {
      method: details.method,
      path: apiInfo.apiPath,
      query: apiInfo.query
    });

    requestMap.set(details.requestId, {
      requestId: details.requestId,
      url: details.url,
      method: details.method,
      tabId: details.tabId,
      timestamp: new Date().toISOString(),
      startTime: Date.now(),
      type: details.type,
      initiator: details.initiator,
      requestBody: details.requestBody
    });
  },
  { urls: SUPABASE_PATTERNS },
  ["requestBody"]
);

// 监听请求头发送
chrome.webRequest.onSendHeaders.addListener(
  (details) => {
    if (requestMap.has(details.requestId)) {
      const request = requestMap.get(details.requestId);
      const headers = {};
      // 保存所有请求头，因为Supabase API需要特定的headers
      if (details.requestHeaders) {
        details.requestHeaders.forEach(header => {
          const name = header.name.toLowerCase();
          // 对敏感header特殊处理
          if (name === 'authorization' || name === 'apikey') {
            headers[name] = header.value.substring(0, 20) + '***';
          } else {
            headers[name] = header.value;
          }
        });
      }
      request.headers = headers;
      requestMap.set(details.requestId, request);
    }
  },
  { urls: SUPABASE_PATTERNS },
  ["requestHeaders"]
);

// 监听响应头接收
chrome.webRequest.onHeadersReceived.addListener(
  (details) => {
    if (requestMap.has(details.requestId)) {
      const request = requestMap.get(details.requestId);
      const responseHeaders = {};
      if (details.responseHeaders) {
        details.responseHeaders.forEach(header => {
          responseHeaders[header.name.toLowerCase()] = header.value;
        });
      }
      request.responseHeaders = responseHeaders;
      requestMap.set(details.requestId, request);

      // 记录响应状态
      const apiInfo = extractApiInfo(details.url);
      console.log(`[Matrix] Supabase ${apiInfo.apiType} 响应状态:`, {
        method: request.method,
        path: apiInfo.apiPath,
        status: details.statusCode,
        contentType: responseHeaders['content-type']
      });
    }
  },
  { urls: SUPABASE_PATTERNS },
  ["responseHeaders"]
);

// 监听响应完成
chrome.webRequest.onCompleted.addListener(
  async (details) => {
    // 处理所有响应，包括成功的200响应
    const isSuccess = details.statusCode >= 200 && details.statusCode < 300;
    console.log(`[Matrix] 捕获到API响应:`, {
      status: details.statusCode,
      url: details.url,
      success: isSuccess
    });
    await handleRequestComplete(details, false, isSuccess);
  },
  { urls: SUPABASE_PATTERNS },
  ["responseHeaders"]
);

// 监听请求错误
chrome.webRequest.onErrorOccurred.addListener(
  async (details) => {
    console.log(`[Matrix] 捕获到请求错误:`, {
      error: details.error,
      url: details.url
    });
    await handleRequestComplete(details, true, false);
  },
  { urls: SUPABASE_PATTERNS }
);

// 格式化请求体
function formatRequestBody(requestBody) {
  if (!requestBody) return null;

  try {
    if (requestBody.formData) {
      const formData = {};
      for (const [key, values] of Object.entries(requestBody.formData)) {
        formData[key] = values.length === 1 ? values[0] : values;
      }
      return formData;
    } else if (requestBody.raw) {
      const decoder = new TextDecoder('utf-8');
      const text = decoder.decode(new Uint8Array(requestBody.raw[0].bytes));
      try {
        return JSON.parse(text);
      } catch {
        return text.length <= 1000 ? text : `[Body size: ${text.length} chars]`;
      }
    }
  } catch (e) {
    return '[Unable to parse body]';
  }
  return null;
}

// 处理请求完成
async function handleRequestComplete(details, isError, isSuccess = false) {
  if (!requestMap.has(details.requestId)) return;

  const request = requestMap.get(details.requestId);
  const duration = Date.now() - request.startTime;
  const apiInfo = extractApiInfo(details.url);

  // 构建日志条目
  const logEntry = {
    type: isError ? 'supabase.api.error' : (isSuccess ? 'supabase.api.success' : 'supabase.api.non200'),
    timestamp: request.timestamp,
    request: {
      projectId: apiInfo.projectId,
      apiType: apiInfo.apiType,
      apiPath: apiInfo.apiPath,
      query: apiInfo.query,
      url: request.url,
      method: request.method,
      headers: request.headers || {},
      body: request.requestBody ? formatRequestBody(request.requestBody) : null,
      initiator: request.initiator
    },
    response: {
      status: details.statusCode,
      statusText: details.statusLine,
      headers: request.responseHeaders || {},
      duration: duration
    },
    success: isSuccess && !isError
  };

  // 如果是错误，添加错误信息
  if (isError) {
    logEntry.error = {
      message: details.error,
      name: 'NetworkError'
    };
  } else if (!isSuccess) {
    logEntry.errorMessage = `HTTP ${details.statusCode}`;
  }

  console.log(`[Matrix] 记录API日志:`, logEntry);

  // 发送日志到对应的标签页
  if (request.tabId > 0) {
    try {
      await chrome.tabs.sendMessage(request.tabId, {
        action: isSuccess ? 'logNetworkSuccess' : 'logNetworkError',
        data: logEntry
      });
    } catch (error) {
      console.log('Failed to send message to tab:', error);
    }
  }

  // 清理请求信息
  requestMap.delete(details.requestId);
}

// 在导航提交时注入脚本
chrome.webNavigation.onCommitted.addListener(async (details) => {
  if (details.frameId === 0) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: details.tabId },
        files: ['content.js'],
        injectImmediately: true,
        world: "MAIN"
      });
    } catch (err) {
      console.error("Early script injection failed:", err);
    }
  }
});

// 注册常规内容脚本作为备份
chrome.runtime.onInstalled.addListener(async () => {
  await chrome.scripting.registerContentScripts([{
    id: "error-logger",
    matches: ["<all_urls>"],
    js: ["content.js"],
    runAt: "document_start",
    world: "MAIN",
    allFrames: true
  }]);
});
