import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  ReactFlow,
  useNodesState, 
  useEdgesState, 
  addEdge, 
  Background, 
  Controls, 
  Handle, 
  Position, 
  MarkerType,
  BaseEdge,
  EdgeLabelRenderer,
  getStraightPath,
  ReactFlowProvider,
  ConnectionMode,
  useReactFlow
} from '@xyflow/react';
import type { Node, Edge, Connection } from '@xyflow/react';
import dagre from 'dagre';
import { LuPlus, LuRefreshCw, LuTrash2, LuNetwork } from 'react-icons/lu';
import '@xyflow/react/dist/style.css';

// --- CUSTOM COMIC NODE ---
const ComicNode = ({ data, selected }: { data: { label: string; type: string; color?: string }; selected: boolean }) => {
  const sizeClass = data.type === 'main' ? 'w-24 h-24 text-sm' : data.type === 'sub' ? 'w-16 h-16 text-xs' : 'w-12 h-12 text-[10px]';
  const bgColor = selected ? 'bg-comic-yellow' : (data.color || 'bg-white');
  
  return (
    <div className={`${sizeClass} relative group`}>
       <div className="absolute inset-0 rounded-full bg-black translate-x-1 translate-y-1" />
       <div className={`
          absolute inset-0 rounded-full border-2 border-black flex items-center justify-center text-center font-black z-10 
          ${bgColor} transition-colors duration-200
       `}>
          {data.label}
       </div>
       <Handle 
         type="source" 
         position={Position.Top} 
         id="top"
         className="w-3 h-3 bg-black border-2 border-white opacity-0 group-hover:opacity-100 transition-opacity z-20" 
       />
       <Handle 
         type="source" 
         position={Position.Bottom} 
         id="bottom"
         className="w-3 h-3 bg-black border-2 border-white opacity-0 group-hover:opacity-100 transition-opacity z-20" 
       />
       <Handle 
         type="source" 
         position={Position.Left} 
         id="left"
         className="w-3 h-3 bg-black border-2 border-white opacity-0 group-hover:opacity-100 transition-opacity z-20" 
       />
       <Handle 
         type="source" 
         position={Position.Right} 
         id="right"
         className="w-3 h-3 bg-black border-2 border-white opacity-0 group-hover:opacity-100 transition-opacity z-20" 
       />
    </div>
  );
};

// --- CUSTOM COMIC EDGE ---
const ComicEdge = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  style = {},
  markerEnd,
  label,
  selected
}: {
  id: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  style?: React.CSSProperties;
  markerEnd?: string;
  label?: string;
  selected?: boolean;
}) => {
  const [edgePath, labelX, labelY] = getStraightPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });

  return (
    <>
      <BaseEdge 
        path={edgePath} 
        markerEnd={markerEnd} 
        style={{ 
            ...style, 
            strokeWidth: 2,
            stroke: selected ? '#F59E0B' : 'black',
            transition: 'stroke 0.2s'
        }} 
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
            className="nodrag nopan"
          >
            <div className={`
                px-2 py-0.5 border border-black rounded-md text-[10px] font-bold shadow-comic-sm transition-colors
                ${selected ? 'bg-comic-yellow text-black' : 'bg-white text-gray-700'}
            `}>
              {label}
            </div>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

const nodeTypes = { comic: ComicNode };
const edgeTypes = { comic: ComicEdge };

// --- INITIAL DATA ---
const initialNodes: Node[] = [
  { id: '1', type: 'comic', position: { x: 250, y: 200 }, data: { label: '李清月', type: 'main', color: 'bg-white' } },
  { id: '2', type: 'comic', position: { x: 100, y: 100 }, data: { label: '林婉儿', type: 'sub', color: 'bg-white' } },
  { id: '3', type: 'comic', position: { x: 400, y: 100 }, data: { label: '顾兵', type: 'sub', color: 'bg-white' } },
];

const initialEdges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', label: '母女', type: 'comic', markerEnd: { type: MarkerType.ArrowClosed, color: 'black' } },
  { id: 'e1-3', source: '1', target: '3', label: '盟友', type: 'comic', markerEnd: { type: MarkerType.ArrowClosed, color: 'black' } },
];

// --- MAIN FLOW COMPONENT ---
const CharacterMapFlow = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const { fitView } = useReactFlow();

    // 监听容器大小变化，自动调整视图
    useEffect(() => {
        if (!containerRef.current) return;
        
        const resizeObserver = new ResizeObserver(() => {
            // 延迟执行 fitView，确保 DOM 已更新
            setTimeout(() => {
                fitView({ padding: 0.2, duration: 200 });
            }, 50);
        });
        
        resizeObserver.observe(containerRef.current);
        
        return () => {
            resizeObserver.disconnect();
        };
    }, [fitView]);

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge({ ...params, type: 'comic', label: 'Relation', markerEnd: { type: MarkerType.ArrowClosed, color: 'black' } }, eds)),
        [setEdges]
    );

    const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
        setSelectedEdge(null);
    }, []);

    const onEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
        setSelectedEdge(edge);
        setSelectedNode(null);
    }, []);

    const onPaneClick = useCallback(() => {
        setSelectedNode(null);
        setSelectedEdge(null);
    }, []);

    const addNode = () => {
        const id = (nodes.length + 1 + Math.random()).toString();
        const newNode: Node = {
            id,
            type: 'comic',
            position: { x: Math.random() * 400 + 50, y: Math.random() * 300 + 50 },
            data: { label: 'New Role', type: 'sub', color: 'bg-white' },
        };
        setNodes((nds) => [...nds, newNode]);
    };

    const onLayout = useCallback((direction = 'TB') => {
        const dagreGraph = new dagre.graphlib.Graph();
        dagreGraph.setDefaultEdgeLabel(() => ({}));
        const nodeWidth = 120;
        const nodeHeight = 120;
        dagreGraph.setGraph({ rankdir: direction });

        nodes.forEach((node) => {
            dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
        });

        edges.forEach((edge) => {
            dagreGraph.setEdge(edge.source, edge.target);
        });

        dagre.layout(dagreGraph);

        const layoutedNodes = nodes.map((node) => {
            const nodeWithPosition = dagreGraph.node(node.id);
            return {
                ...node,
                position: {
                    x: nodeWithPosition.x - nodeWidth / 2,
                    y: nodeWithPosition.y - nodeHeight / 2,
                },
            };
        });

        setNodes(layoutedNodes);
    }, [nodes, edges, setNodes]);

    const updateNodeData = (key: string, value: string) => {
        if (!selectedNode) return;
        setNodes((nds) =>
            nds.map((n) => {
                if (n.id === selectedNode.id) {
                    const updated = { ...n, data: { ...n.data, [key]: value } };
                    setSelectedNode(updated); 
                    return updated;
                }
                return n;
            })
        );
    };

    const updateEdgeLabel = (newLabel: string) => {
        if (!selectedEdge) return;
        setEdges((eds) =>
            eds.map((e) => {
                if (e.id === selectedEdge.id) {
                    const updated = { ...e, label: newLabel };
                    setSelectedEdge(updated);
                    return updated;
                }
                return e;
            })
        );
    };

    const deleteSelection = () => {
        if (selectedNode) {
            setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
            setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
            setSelectedNode(null);
        } else if (selectedEdge) {
            setEdges((eds) => eds.filter((e) => e.id !== selectedEdge.id));
            setSelectedEdge(null);
        }
    };

    return (
        <div ref={containerRef} className="w-full h-full flex flex-col relative">
            {/* Header */}
            <div className="h-14 border-b-4 border-black bg-comic-yellow flex items-center justify-between px-4 shrink-0 z-20">
                <div className="flex items-center gap-3">
                    <LuNetwork size={24} className="text-black" />
                    <h2 className="font-black text-lg uppercase tracking-wider">tailmemo</h2>
                </div>
            </div>

            <div className="flex-1 relative flex overflow-hidden">
                {/* Canvas */}
                <div className="flex-1 h-full">
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onConnect={onConnect}
                        onNodeClick={onNodeClick}
                        onEdgeClick={onEdgeClick}
                        onPaneClick={onPaneClick}
                        nodeTypes={nodeTypes}
                        edgeTypes={edgeTypes}
                        fitView
                        attributionPosition="bottom-left"
                        defaultEdgeOptions={{ type: 'comic' }}
                        connectionMode={ConnectionMode.Loose}
                    >
                        <Background color="#aaa" gap={20} size={1} />
                        <Controls className="bg-white border-2 border-black shadow-comic-sm" />
                    </ReactFlow>
                </div>

                {/* Floating Toolbar (Left) */}
                <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
                     <button onClick={addNode} className="w-10 h-10 bg-white border-2 border-black shadow-comic flex items-center justify-center hover:bg-gray-50 active:translate-y-0.5 active:shadow-sm transition-all" title="Add Character">
                         <LuPlus size={20} />
                     </button>
                     <button onClick={() => onLayout('TB')} className="w-10 h-10 bg-white border-2 border-black shadow-comic flex items-center justify-center hover:bg-gray-50 active:translate-y-0.5 active:shadow-sm transition-all" title="Auto Layout">
                         <LuRefreshCw size={18} />
                     </button>
                </div>

                {/* Inspector Panel (Right) */}
                {(selectedNode || selectedEdge) && (
                    <div className="w-64 bg-white border-l-4 border-black p-4 flex flex-col gap-4 z-20 shadow-[-4px_0px_0px_0px_rgba(0,0,0,0.1)]">
                        <div className="flex items-center justify-between">
                            <h3 className="font-black text-sm uppercase">
                                {selectedNode ? 'Edit Character' : 'Edit Relation'}
                            </h3>
                            <button onClick={deleteSelection} className="text-red-500 hover:bg-red-50 p-1 rounded transition-colors" title="Delete">
                                <LuTrash2 size={16}/>
                            </button>
                        </div>
                        
                        {selectedNode && (
                            <>
                                <div className="space-y-1">
                                    <label className="text-[10px] font-bold text-gray-500 uppercase">Name</label>
                                    <input 
                                        type="text" 
                                        value={selectedNode.data.label as string}
                                        onChange={(e) => updateNodeData('label', e.target.value)}
                                        className="w-full border-2 border-black p-2 font-bold text-sm outline-none focus:bg-yellow-50 focus:shadow-comic-sm transition-all"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] font-bold text-gray-500 uppercase">Role Importance</label>
                                    <div className="flex gap-2">
                                        {['main', 'sub', 'minor'].map((type) => (
                                            <button 
                                                key={type}
                                                onClick={() => updateNodeData('type', type)}
                                                className={`flex-1 py-1 border-2 text-[10px] font-bold uppercase transition-all
                                                    ${selectedNode.data.type === type 
                                                        ? 'border-black bg-black text-white shadow-sm' 
                                                        : 'border-gray-200 text-gray-400 hover:border-black hover:text-black'
                                                    }
                                                `}
                                            >
                                                {type}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] font-bold text-gray-500 uppercase">Color Tag</label>
                                    <div className="flex gap-2">
                                        {['bg-white', 'bg-comic-yellow', 'bg-blue-100', 'bg-red-100'].map((color) => (
                                            <button 
                                                key={color}
                                                onClick={() => updateNodeData('color', color)}
                                                className={`w-8 h-8 rounded-full border-2 border-black ${color} hover:scale-110 transition-transform ${selectedNode.data.color === color ? 'ring-2 ring-offset-2 ring-black' : ''}`}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </>
                        )}

                        {selectedEdge && (
                             <div className="space-y-1">
                                <label className="text-[10px] font-bold text-gray-500 uppercase">Relationship Label</label>
                                <input 
                                    type="text" 
                                    value={selectedEdge.label as string || ''}
                                    onChange={(e) => updateEdgeLabel(e.target.value)}
                                    className="w-full border-2 border-black p-2 font-bold text-sm outline-none focus:bg-yellow-50 focus:shadow-comic-sm transition-all"
                                    placeholder="e.g. Friends, Enemy"
                                    autoFocus
                                />
                                <p className="text-[10px] text-gray-400 mt-1">
                                    Describe how these two characters are connected.
                                </p>
                            </div>
                        )}
                        
                        <div className="mt-auto p-3 bg-gray-50 border-2 border-black text-[10px] text-gray-500 font-medium leading-tight">
                            Tip: Select a line to rename it. Press the trash icon to delete nodes or connections.
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// --- FULL PAGE COMPONENT (主页使用) ---
export const CharacterMapPage: React.FC = () => {
  return (
    <div className="w-full h-full">
      <ReactFlowProvider>
        <CharacterMapFlow />
      </ReactFlowProvider>
    </div>
  );
};
