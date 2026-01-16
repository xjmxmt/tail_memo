import React, { useState, useRef, useEffect } from 'react';
import { LuSend, LuMessageSquare, LuLoader, LuX } from 'react-icons/lu';

// API 配置
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface MemoryItem {
  content: string;
  source?: string;
  similarity?: number;
}

interface Message {
  id: number;
  type: 'user' | 'system';
  content: string;
  memories?: MemoryItem[];
  isLoading?: boolean;
}

interface QueryResponse {
  reply: string;
  memories: MemoryItem[];
}

interface ChatPanelProps {
  onClose: () => void;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({ onClose }) => {
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      type: 'system',
      content: '欢迎使用记忆查询系统！输入人物名称或关键词来查找相关记忆。',
    },
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: messages.length,
      type: 'user',
      content: inputValue,
    };

    // 添加用户消息和加载中状态
    const loadingMessage: Message = {
      id: messages.length + 1,
      type: 'system',
      content: '正在查询...',
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputValue }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: QueryResponse = await response.json();

      // 替换加载消息为实际响应
      const systemMessage: Message = {
        id: messages.length + 1,
        type: 'system',
        content: data.reply,
        memories: data.memories,
      };

      setMessages((prev) => [...prev.slice(0, -1), systemMessage]);
    } catch (error) {
      console.error('查询失败:', error);
      
      // 替换加载消息为错误消息
      const errorMessage: Message = {
        id: messages.length + 1,
        type: 'system',
        content: '查询失败，请检查后端服务是否启动。',
      };

      setMessages((prev) => [...prev.slice(0, -1), errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="h-14 bg-comic-yellow border-b-4 border-black flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-2">
          <LuMessageSquare size={20} className="text-black" />
          <span className="font-black text-lg uppercase tracking-wider">Memory Query</span>
        </div>
        <button 
          onClick={onClose}
          className="w-8 h-8 flex items-center justify-center bg-white border-2 border-black hover:bg-red-500 hover:text-white transition-colors shadow-comic-sm"
        >
          <LuX size={18} strokeWidth={3} />
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] ${
                msg.type === 'user'
                  ? 'bg-comic-yellow border-2 border-black shadow-comic-sm'
                  : 'bg-white border-2 border-black shadow-comic-sm'
              } px-3 py-2`}
            >
              {msg.isLoading ? (
                <div className="flex items-center gap-2 text-sm font-medium text-gray-500">
                  <LuLoader size={16} className="animate-spin" />
                  {msg.content}
                </div>
              ) : (
                <p className="text-sm font-medium">{msg.content}</p>
              )}
              
              {/* Memory List */}
              {msg.memories && msg.memories.length > 0 && (
                <div className="mt-2 space-y-1">
                  {msg.memories.map((memory, idx) => (
                    <div
                      key={idx}
                      className="text-xs bg-blue-50 border border-blue-200 px-2 py-1 rounded"
                    >
                      <span>💭 {memory.content}</span>
                      {memory.source && (
                        <span className="ml-2 text-gray-400">— {memory.source}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {/* 滚动锚点 */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t-4 border-black p-4 bg-white shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入人物名称或关键词..."
            disabled={isLoading}
            className="flex-1 border-2 border-black px-3 py-2 text-sm font-medium outline-none focus:bg-yellow-50 focus:shadow-comic-sm transition-all disabled:bg-gray-100 disabled:text-gray-400"
          />
          <button
            onClick={handleSend}
            disabled={isLoading}
            className="w-10 h-10 bg-comic-yellow border-2 border-black shadow-comic-sm flex items-center justify-center hover:bg-yellow-300 active:translate-y-0.5 active:shadow-none transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
          >
            {isLoading ? (
              <LuLoader size={18} className="animate-spin" />
            ) : (
              <LuSend size={18} />
            )}
          </button>
        </div>
        <p className="text-[10px] text-gray-400 mt-2">
          提示：试试输入 "李清月"、"林婉儿" 或 "顾兵"
        </p>
      </div>
    </div>
  );
};
