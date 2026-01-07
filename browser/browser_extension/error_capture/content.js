// 幂等的错误捕获实现
(function () {
  // 如果已经初始化过，直接返回
  if (window.__matrix_errors_initialized__) return;
  window.__matrix_errors_initialized__ = true;

  // 初始化错误存储数组
  if (!window.__matrix_errors__) {
    window.__matrix_errors__ = [];
  }

  // 初始化成功API响应存储数组
  if (!window.__matrix_api_success__) {
    window.__matrix_api_success__ = [];
  }

  // 数据截断配置
  const TRUNCATE_CONFIG = {
    maxStringLength: 1000,
    maxArrayLength: 50,
    maxObjectKeys: 20,
    maxStackLines: 20,
  };

  // 数据截断工具函数
  function truncateData(data, depth = 0) {
    if (depth > 3) return '[Max Depth Exceeded]';

    if (typeof data === 'string') {
      if (data.length > TRUNCATE_CONFIG.maxStringLength) {
        return data.substring(0, TRUNCATE_CONFIG.maxStringLength) + `... [truncated ${data.length - TRUNCATE_CONFIG.maxStringLength} chars]`;
      }
      return data;
    }

    if (Array.isArray(data)) {
      if (data.length > TRUNCATE_CONFIG.maxArrayLength) {
        return data
          .slice(0, TRUNCATE_CONFIG.maxArrayLength)
          .map(item => truncateData(item, depth + 1))
          .concat([`... [truncated ${data.length - TRUNCATE_CONFIG.maxArrayLength} items]`]);
      }
      return data.map(item => truncateData(item, depth + 1));
    }

    if (data && typeof data === 'object') {
      const keys = Object.keys(data);
      if (keys.length > TRUNCATE_CONFIG.maxObjectKeys) {
        const truncatedObj = {};
        keys.slice(0, TRUNCATE_CONFIG.maxObjectKeys).forEach(key => {
          truncatedObj[key] = truncateData(data[key], depth + 1);
        });
        truncatedObj['__truncated'] = `[${keys.length - TRUNCATE_CONFIG.maxObjectKeys} more fields]`;
        return truncatedObj;
      }
      const processedObj = {};
      keys.forEach(key => {
        processedObj[key] = truncateData(data[key], depth + 1);
      });
      return processedObj;
    }

    return data;
  }

  // 处理错误堆栈
  function truncateStack(stack) {
    if (!stack) return null;
    const lines = stack.split('\n');
    if (lines.length > TRUNCATE_CONFIG.maxStackLines) {
      return lines
        .slice(0, TRUNCATE_CONFIG.maxStackLines)
        .concat([`... [truncated ${lines.length - TRUNCATE_CONFIG.maxStackLines} stack lines]`])
        .join('\n');
    }
    return stack;
  }

  // 安全地记录错误
  function safeLogError(error) {
    if (!window.__matrix_errors__) {
      window.__matrix_errors__ = [];
    }
    // 限制数组大小
    if (window.__matrix_errors__.length >= 1000) {
      window.__matrix_errors__.shift(); // 移除最旧的错误
    }
    window.__matrix_errors__.push(truncateData(error));
  }

  // 安全地记录成功的API响应
  function safeLogApiSuccess(apiResponse) {
    if (!window.__matrix_api_success__) {
      window.__matrix_api_success__ = [];
    }
    // 限制数组大小
    if (window.__matrix_api_success__.length >= 500) {
      window.__matrix_api_success__.shift(); // 移除最旧的记录
    }
    window.__matrix_api_success__.push(truncateData(apiResponse));
  }

  // 保存原始console方法（如果尚未保存）
  if (!window.__original_console_error__) {
    window.__original_console_error__ = console.error;
  }

  if (!window.__original_console_log__) {
    window.__original_console_log__ = console.log;
  }

  // 监听来自injector.js的消息
  window.addEventListener('message', function (event) {
    // 确保消息来源安全且类型正确
    if (event.source === window && event.data) {
      if (event.data.type === 'MATRIX_ERROR_LOG') {
        safeLogError(event.data.data);
      } else if (event.data.type === 'MATRIX_API_SUCCESS_LOG') {
        safeLogApiSuccess(event.data.data);
      }
    }
  });

  // 覆盖console.error
  console.error = function (...args) {
    safeLogError({
      type: 'console.error',
      message: truncateData(args.join(' ')),
      timestamp: new Date().toISOString(),
      stack: truncateStack(new Error().stack)
    });
    window.__original_console_error__.apply(console, args);
  };

  // 覆盖console.log
  console.log = function (...args) {
    safeLogError({
      type: 'console.log',
      message: truncateData(args.join(' ')),
      timestamp: new Date().toISOString()
    });
    window.__original_console_log__.apply(console, args);
  };

  // 捕获图片加载失败事件
  document.addEventListener('error', function (event) {
    if (event.target.tagName === 'IMG') {
      safeLogError({
        type: 'image.error',
        message: `Failed to load image: ${event.target.src}`,
        timestamp: new Date().toISOString(),
        stack: truncateStack(new Error().stack),
        element: truncateData({
          tagName: event.target.tagName,
          src: event.target.src,
          id: event.target.id,
          className: event.target.className
        })
      });
    }
  }, true);

  // 捕获未处理的错误
  window.addEventListener('error', function (event) {
    safeLogError({
      type: 'uncaught.error',
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      timestamp: new Date().toISOString(),
      stack: truncateStack(event.error ? event.error.stack : null)
    });
    return false;
  }, true);

  // 捕获未处理的Promise拒绝
  window.addEventListener('unhandledrejection', function (event) {
    let message = 'Promise rejection: ';
    if (typeof event.reason === 'object') {
      message += truncateData(event.reason.message || JSON.stringify(event.reason));
    } else {
      message += truncateData(String(event.reason));
    }

    safeLogError({
      type: 'unhandled.promise',
      message: message,
      timestamp: new Date().toISOString(),
      stack: truncateStack(event.reason && event.reason.stack ? event.reason.stack : null)
    });
  });
})();
