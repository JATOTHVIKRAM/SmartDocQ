import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FileText, MessageSquare, User, Bot, Send, Upload, Plus, X, MoreVertical, LogOut } from "lucide-react";
import FileUpload from "@/components/file-upload";
import Navigation from "@/components/navigation";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

interface Document {
  id: string;
  name: string;
  preview: string;
  uploadDate: Date;
  isActive: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  documentId: string;
  documentName?: string;
  messageCount: number;
}

interface SmartDocumentationProps {
  isAuthenticated: boolean;
  userEmail: string;
  onLogout: () => void;
}

export default function SmartDocumentation({ 
  isAuthenticated, 
  userEmail, 
  onLogout 
}: SmartDocumentationProps) {
  const API_BASE = "http://127.0.0.1:8000/api";
  const [documents, setDocuments] = useState<Document[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState<string>("");
  const [showUpload, setShowUpload] = useState<boolean>(false);
  const [isUploadingDoc, setIsUploadingDoc] = useState<boolean>(false);
  const [isSendingMessage, setIsSendingMessage] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [docLoaded, setDocLoaded] = useState<boolean>(false);
  const [authToken, setAuthToken] = useState<string>("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Get auth token on mount
  useEffect(() => {
    getAuthToken();
  }, []);

  // Once authToken is available/changes, load chat sessions
  useEffect(() => {
    if (authToken) {
      loadChatSessions();
    }
  }, [authToken]);

  // Get authentication token from localStorage
  const getAuthToken = (): void => {
    const token = localStorage.getItem("token") || localStorage.getItem("authToken");
    setAuthToken(token || "");
  };

  // Load chat sessions from backend
  const loadChatSessions = async (): Promise<void> => {
    try {
      // Read latest token to avoid stale state on first render
      const latestToken = localStorage.getItem("token") || localStorage.getItem("authToken") || authToken;
      const headers: HeadersInit = {
        "Content-Type": "application/json",
      };
      
      if (latestToken) {
        headers["Authorization"] = `Bearer ${latestToken}`;
      }
      
      const response = await fetch(`${API_BASE}/chat/sessions`, { headers });
      if (response.ok) {
        const sessions = await response.json();
        const formattedSessions: ChatSession[] = sessions.map((session: any) => ({
          id: session.id,
          title: session.title,
          lastMessage: session.last_message || "",
          timestamp: new Date(session.updated_at),
          documentId: "1", // Default document ID
          documentName: session.document_name,
          messageCount: session.message_count || 0
        }));
        setChatSessions(formattedSessions);
      }
    } catch (err) {
      console.error("Failed to load chat sessions:", err);
    }
  };

  // Load specific chat session messages
  const loadChatSession = async (sessionId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`);
      if (response.ok) {
        const session = await response.json();
        const formattedMessages: Message[] = session.messages.map((msg: any) => ({
          id: msg.id,
          type: msg.type,
          content: msg.content,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(formattedMessages);
      }
    } catch (err) {
      console.error("Failed to load chat session:", err);
    }
  };

  // Logout
  const handleLogout = async (): Promise<void> => {
    try {
      await fetch(`${API_BASE}/logout`, { method: "POST" });
      setDocuments([]);
      setChatSessions([]);
      setMessages([]);
      setCurrentChatId("");
      setDocLoaded(false);
      setAuthToken("");
      // Call parent logout function to update global auth state
      onLogout();
      console.log("Logged out and cache cleared");
    } catch (err) {
      console.error("Logout failed:", err);
    }
  };

  // Upload
  const handleFileUpload = async (file: File): Promise<void> => {
    if (!file.name.toLowerCase().match(/\.(pdf|docx|txt)$/)) {
      setError("Only PDF, DOCX, and TXT files are supported.");
      return;
    }

    setIsUploadingDoc(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      console.log("Document processed:", data);

      setDocuments([
        {
          id: Date.now().toString(),
          name: file.name,
          preview: "Document ready for Q&A",
          uploadDate: new Date(),
          isActive: true,
        },
      ]);
      setDocLoaded(true);
      setShowUpload(false);

      // Start a new chat after document upload
      handleNewChat();
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setIsUploadingDoc(false);
    }
  };

  // Send message
  const handleSendMessage = async (): Promise<void> => {
    if (!inputValue.trim()) return;

    if (!docLoaded) {
      setError("⚠️ Please upload and process a document first.");
      return;
    }

    const newMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputValue("");
    setIsSendingMessage(true);
    setError(null);

    try {
      const headers: HeadersInit = {
        "Content-Type": "application/json",
      };
      
      if (authToken) {
        headers["Authorization"] = `Bearer ${authToken}`;
      }
      
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify({ 
          question: newMessage.content,
          chat_session_id: currentChatId || undefined
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text);
      }

      const data = await response.json();
      
      // Update current chat ID if this is a new session
      if (data.chat_session_id && !currentChatId) {
        setCurrentChatId(data.chat_session_id);
        // Reload chat sessions to include the new one
        loadChatSessions();
      }
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: data.answer || "No answer received.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err: any) {
      const msg = err.message || "Error contacting model.";
      setError(msg);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          type: "bot",
          content: `❌ ${msg}`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsSendingMessage(false);
    }
  };

  const handleNewChat = async (): Promise<void> => {
    try {
      const headers: HeadersInit = {
        "Content-Type": "application/json",
      };
      
      if (authToken) {
        headers["Authorization"] = `Bearer ${authToken}`;
      }
      
      const response = await fetch(`${API_BASE}/chat/sessions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          title: `New Chat ${new Date().toLocaleString()}`,
          document_name: documents[0]?.name
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentChatId(data.session_id);
        setMessages([]);
        // Reload chat sessions to include the new one
        loadChatSessions();
      }
    } catch (err) {
      console.error("Failed to create new chat:", err);
      // Fallback to local creation
      const newChat: ChatSession = {
        id: Date.now().toString(),
        title: "New Chat",
        lastMessage: "",
        timestamp: new Date(),
        documentId: documents[0]?.id || "1",
        messageCount: 0
      };
      setChatSessions((prev) => [newChat, ...prev]);
      setCurrentChatId(newChat.id);
      setMessages([]);
    }
  };

  const switchChat = async (chatId: string): Promise<void> => {
    setCurrentChatId(chatId);
    await loadChatSession(chatId);
  };

  const currentChat = chatSessions.find((chat) => chat.id === currentChatId);

  return (
    <>
      <Navigation 
        onAuthModal={() => {}} 
        isAuthenticated={isAuthenticated} 
        userEmail={userEmail}
        onLogout={handleLogout}
      />
      <div className="flex h-screen bg-primary-bg pt-16">
        {/* Sidebar */}
        <div className="w-80 bg-card-bg border-r border-border-color flex flex-col">
          <div className="p-4 border-b border-border-color">
            <Button onClick={handleNewChat} className="w-full bg-primary-blue hover:bg-blue-600 mb-3">
              <Plus className="h-4 w-4 mr-2" /> New Chat
            </Button>
            <Button
              onClick={() => setShowUpload(!showUpload)}
              variant="outline"
              className="w-full border-border-color"
            >
              <Upload className="h-4 w-4 mr-2" /> Upload Document
            </Button>
          </div>

          {showUpload && (
            <div className="p-4 border-b border-border-color">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Upload Document</h3>
                  <Button variant="ghost" size="sm" onClick={() => setShowUpload(false)}>
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                {error && <p className="text-red-500 text-xs">{error}</p>}
                {isUploadingDoc ? (
                  <div className="text-center py-4 text-sm">Processing...</div>
                ) : (
                  <FileUpload
                    onFileSelect={handleFileUpload}
                    acceptedTypes=".pdf,.docx,.txt"
                    maxSize={10}
                    className="min-h-[100px]"
                  />
                )}
              </div>
            </div>
          )}

          {/* Documents Section */}
          <div className="p-4 border-b border-border-color">
            <h3 className="text-sm font-medium text-text-secondary mb-3">Documents</h3>
            {documents.length === 0 ? (
              <p className="text-xs text-text-secondary">No document uploaded yet.</p>
            ) : (
              documents.map((doc) => (
                <div
                  key={doc.id}
                  className={`flex items-center p-2 rounded-lg cursor-pointer transition-colors ${
                    doc.isActive ? "bg-primary-blue/10 border border-primary-blue/30" : "hover:bg-border-color/20"
                  }`}
                >
                  <FileText className={`mr-3 h-4 w-4 ${doc.isActive ? "text-primary-blue" : "text-text-secondary"}`} />
                  <div className="flex-1 min-w-0">
                    <div
                      className={`font-medium text-xs truncate ${doc.isActive ? "text-primary-blue" : ""}`}
                    >
                      {doc.name}
                    </div>
                    <div className="text-xs text-text-secondary truncate">{doc.preview}</div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-medium text-text-secondary mb-3">Recent Chats</h3>
            {chatSessions.map((chat) => (
              <div
                key={chat.id}
                onClick={() => switchChat(chat.id)}
                className={`p-3 rounded-lg cursor-pointer ${
                  chat.id === currentChatId
                    ? "bg-primary-blue/10 border border-primary-blue/30"
                    : "hover:bg-border-color/20"
                }`}
              >
                <div
                  className={`font-medium text-sm truncate mb-1 ${
                    chat.id === currentChatId ? "text-primary-blue" : ""
                  }`}
                >
                  {chat.title}
                </div>
                <div className="text-xs text-text-secondary truncate">
                  {chat.lastMessage || "No messages yet"}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Main Chat Interface */}
        <div className="flex-1 flex flex-col">
          <div className="border-b border-border-color p-4 flex justify-between items-center">
            <div className="flex items-center">
              <MessageSquare className="text-primary-blue h-6 w-6 mr-3" />
              <div>
                <h2 className="font-semibold">{currentChat?.title || "Smart Documentation"}</h2>
                <p className="text-sm text-text-secondary">
                  {documents.find((d) => d.isActive)?.name || "No document selected"}
                </p>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex-1 p-6 overflow-y-auto">
            {messages.length === 0 ? (
              <div className="text-center text-text-secondary">
                Upload a document and ask a question to begin.
              </div>
            ) : (
              <div className="max-w-4xl mx-auto space-y-6">
                {messages.map((message) => (
                  <div key={message.id} className="flex items-start space-x-4">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        message.type === "user"
                          ? "bg-primary-blue text-white"
                          : "bg-card-bg border border-border-color text-primary-blue"
                      }`}
                    >
                      {message.type === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                    </div>
                    <div className="flex-1">
                      <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                      <div className="text-xs text-text-secondary mt-2">
                        {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <div className="border-t border-border-color p-4">
            <div className="relative max-w-4xl mx-auto">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                placeholder="Ask a question about your document..."
                disabled={isSendingMessage}
                className="w-full bg-card-bg border-border-color pr-12 focus:border-primary-blue"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isSendingMessage}
                variant="ghost"
                size="sm"
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-primary-blue"
              >
                {isSendingMessage ? (
                  <div className="w-4 h-4 border-2 border-primary-blue border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
            {error && <p className="text-red-500 text-xs mt-2 text-center">{error}</p>}
          </div>
        </div>
      </div>
    </>
  );
}
