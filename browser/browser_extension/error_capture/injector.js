// 这个脚本运行在ISOLATED world中，可以访问chrome.runtime API
// 它的作用是接收来自background script的消息，并转发到MAIN world

// 监听来自background script的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // 处理所有网络相关的消息（成功和错误）
  if ((message.action === 'logNetworkError' || message.action === 'logNetworkSuccess') && message.data) {
    // 使用postMessage将数据传递到MAIN world
    window.postMessage({
      type: message.action === 'logNetworkSuccess' ? 'MATRIX_API_SUCCESS_LOG' : 'MATRIX_ERROR_LOG',
      data: message.data
    }, '*');
  }
  // 发送响应，表示消息已处理
  sendResponse({ received: true });
  return true;
});
