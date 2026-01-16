import { useState, useCallback, useRef, useEffect } from 'react'
import { CharacterMapPage } from './components/CharacterMapModal'
import { ChatPanel } from './components/ChatDialog'
import { LuMessageSquare } from 'react-icons/lu'

function App() {
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [splitPosition, setSplitPosition] = useState(66.67) // 左侧面板宽度百分比
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return
    
    const containerRect = containerRef.current.getBoundingClientRect()
    const newPosition = ((e.clientX - containerRect.left) / containerRect.width) * 100
    
    // 限制范围：最小 30%，最大 80%
    const clampedPosition = Math.min(Math.max(newPosition, 30), 80)
    setSplitPosition(clampedPosition)
  }, [isDragging])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  return (
    <div 
      ref={containerRef}
      className="w-full h-screen bg-gray-100 flex relative border-4 border-black box-border"
    >
      {/* 左侧：人物关系图 */}
      <div 
        className="h-full overflow-hidden"
        style={{ width: isChatOpen ? `${splitPosition}%` : '100%' }}
      >
        <CharacterMapPage />
      </div>
      
      {/* 可拖动分隔条 */}
      {isChatOpen && (
        <div
          onMouseDown={handleMouseDown}
          className={`w-1 h-full bg-black cursor-col-resize hover:bg-black/50 transition-colors shrink-0 ${
            isDragging ? 'bg-black/50' : ''
          }`}
        />
      )}

      {/* 右侧：聊天面板 */}
      <div 
        className={`h-full bg-white overflow-hidden ${isChatOpen ? '' : 'w-0'}`}
        style={{ width: isChatOpen ? `${100 - splitPosition}%` : 0 }}
      >
        {isChatOpen && <ChatPanel onClose={() => setIsChatOpen(false)} />}
      </div>

      {/* 悬浮按钮：打开聊天 */}
      {!isChatOpen && (
        <button
          onClick={() => setIsChatOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-comic-yellow border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-0.5 transition-all z-50"
          title="打开记忆查询"
        >
          <LuMessageSquare size={24} className="text-black" />
        </button>
      )}
    </div>
  )
}

export default App
