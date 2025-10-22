// import { useState, useEffect } from "react";
// import { Button } from "@/components/ui/button";
// import { Input } from "@/components/ui/input";
// import {
//   Select,
//   SelectTrigger,
//   SelectContent,
//   SelectItem,
//   SelectValue,
// } from "@/components/ui/select";
// import { MessageSquare, User, Bot, Send, Upload, FileText, Settings } from "lucide-react";
// import FileUpload from "@/components/file-upload";
// import Navigation from "@/components/navigation";

// interface Message {
//   id: string;
//   type: "user" | "bot";
//   content: string;
//   timestamp: Date;
// }

// export default function InterviewCopilot() {
//   const [messages, setMessages] = useState<Message[]>([]);
//   const [inputValue, setInputValue] = useState("");
//   const [showUpload, setShowUpload] = useState(false);
//   const [uploadedDoc, setUploadedDoc] = useState<File | null>(null);
//   const [numQuestions, setNumQuestions] = useState<number>(5);
//   const [questionType, setQuestionType] = useState<string>("technical");
//   const [questions, setQuestions] = useState<string[]>([]);
//   const [responses, setResponses] = useState<string[]>([]);
//   const [sessionId, setSessionId] = useState<string | null>(null);
//   const [feedback, setFeedback] = useState<string | null>(null);
//   const [avgScore, setAvgScore] = useState<number | null>(null);
//   const [userEmail, setUserEmail] = useState<string>("User");

//   // Map frontend question types to backend levels
//   const typeToLevelMap: Record<string, string> = {
//     technical: "medium",
//     mcq: "easy",
//     hr: "easy",
//     theory: "hard",
//   };

//   // ✅ Load user info and JWT token from localStorage
//   const token = localStorage.getItem("token");
//   useEffect(() => {
//     const email = localStorage.getItem("userEmail");
//     if (email) setUserEmail(email);
//   }, []);

//   const handleFileUpload = (file: File) => {
//     setUploadedDoc(file);
//     setShowUpload(false);
//   };

//   // Start interview
//   const startInterview = async () => {
//     if (!uploadedDoc) {
//       alert("Please upload a document.");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", uploadedDoc);
//     formData.append("num_questions", numQuestions.toString());
//     formData.append("level", typeToLevelMap[questionType] || "medium");

//     try {
//       const response = await fetch("http://127.0.0.1:8000/api/interview/start", {
//         method: "POST",
//         body: formData,
//         headers: {
//           ...(token ? { Authorization: `Bearer ${token}` } : {}),
//         },
//       });

//       if (!response.ok) {
//         const errorText = await response.text();
//         throw new Error(`Error ${response.status}: ${errorText}`);
//       }

//       const data = await response.json();

//       if (!data.questions || data.questions.length === 0) {
//         throw new Error("No questions generated");
//       }

//       setQuestions(data.questions);
//       setResponses(new Array(data.questions.length).fill(""));
//       setSessionId(data.session_id);

//       // Display questions as bot messages
//       const botMessage: Message = {
//         id: Date.now().toString(),
//         type: "bot",
//         content: data.questions.map((q: string, idx: number) => `${idx + 1}. ${q}`).join("\n\n"),
//         timestamp: new Date(),
//       };
//       setMessages((prev) => [...prev, botMessage]);
//     } catch (error: any) {
//       console.error("Error starting interview:", error);
//       alert(error?.message || "Failed to start interview");
//     }
//   };

//   // Update answer for a question
//   const updateAnswer = (index: number, value: string) => {
//     const updated = [...responses];
//     updated[index] = value;
//     setResponses(updated);
//   };

//   // Submit all answers
//   const submitAnswers = async () => {
//     if (!sessionId) return;

//     try {
//       const response = await fetch(`http://127.0.0.1:8000/api/interview/${sessionId}/submit`, {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           ...(token ? { Authorization: `Bearer ${token}` } : {}),
//         },
//         body: JSON.stringify({ responses }),
//       });

//       if (!response.ok) {
//         const errorText = await response.text();
//         throw new Error(`Error ${response.status}: ${errorText}`);
//       }

//       const data = await response.json();
//       setFeedback(data.feedback);
//       setAvgScore(data.avg_score);

//       const botMessage: Message = {
//         id: (Date.now() + 1).toString(),
//         type: "bot",
//         content: `✅ Interview Completed!\nAverage Score: ${data.avg_score}\n\nFeedback:\n${data.feedback}`,
//         timestamp: new Date(),
//       };
//       setMessages((prev) => [...prev, botMessage]);
//     } catch (error: any) {
//       console.error("Error submitting answers:", error);
//       alert(error?.message || "Failed to submit answers");
//     }
//   };

//   return (
//     <>
//       <Navigation 
//         onAuthModal={() => {}} 
//         isAuthenticated={isAuthenticated} 
//         userEmail={userEmail}
//         onLogout={onLogout}
//       />
//       <div className="flex h-screen bg-primary-bg pt-16">
//         {/* Sidebar */}
//         <div className="w-80 bg-card-bg border-r border-border-color flex flex-col">
//           <div className="p-4 border-b border-border-color">
//             <Button
//               onClick={() => setShowUpload(!showUpload)}
//               className="w-full bg-primary-blue hover:bg-blue-600 mb-3"
//             >
//               <Upload className="h-4 w-4 mr-2" />
//               Upload Document
//             </Button>
//           </div>

//           {showUpload && (
//             <div className="p-4 border-b border-border-color">
//               <FileUpload
//                 onFileSelect={handleFileUpload}
//                 acceptedTypes=".pdf,.docx,.txt"
//                 maxSize={10}
//                 className="min-h-[100px]"
//               />
//             </div>
//           )}

//           <div className="p-4 border-b border-border-color">
//             <h3 className="text-sm font-medium mb-3 flex items-center">
//               <Settings className="h-4 w-4 mr-2" /> Question Settings
//             </h3>
//             <div className="space-y-3">
//               <div>
//                 <label className="text-xs text-text-secondary">Number of Questions</label>
//                 <Input
//                   type="number"
//                   min={1}
//                   max={50}
//                   value={numQuestions}
//                   onChange={(e) => setNumQuestions(Number(e.target.value))}
//                 />
//               </div>
//               <div>
//                 <label className="text-xs text-text-secondary">Question Type</label>
//                 <Select onValueChange={setQuestionType} value={questionType}>
//                   <SelectTrigger>
//                     <SelectValue placeholder="Select type" />
//                   </SelectTrigger>
//                   <SelectContent>
//                     <SelectItem value="technical">Technical</SelectItem>
//                     <SelectItem value="mcq">MCQ</SelectItem>
//                     <SelectItem value="hr">HR</SelectItem>
//                     <SelectItem value="theory">Theory</SelectItem>
//                   </SelectContent>
//                 </Select>
//               </div>
//             </div>
//           </div>

//           <div className="p-4">
//             {!sessionId ? (
//               <Button onClick={startInterview} className="w-full bg-primary-blue hover:bg-blue-600">
//                 Start Interview
//               </Button>
//             ) : (
//               <Button onClick={submitAnswers} className="w-full bg-green-600 hover:bg-green-700">
//                 Submit Answers
//               </Button>
//             )}
//           </div>
//         </div>

//         {/* Chat Section */}
//         <div className="flex-1 flex flex-col">
//           <div className="border-b border-border-color p-4 flex items-center justify-between">
//             <MessageSquare className="text-primary-blue h-6 w-6 mr-3" />
//             <h2 className="font-semibold">Interview Copilot</h2>
//             <span className="text-sm text-text-secondary">Logged in as {userEmail}</span>
//           </div>

//           <div className="flex-1 p-6 overflow-y-auto">
//             {messages.length === 0 ? (
//               <div className="flex flex-col items-center justify-center h-full text-center text-text-secondary">
//                 <FileText className="h-12 w-12 mb-3 text-primary-blue" />
//                 <p>Upload a document and I’ll generate interview questions for you.</p>
//               </div>
//             ) : (
//               <div className="max-w-3xl mx-auto space-y-6">
//                 {messages.map((message) => (
//                   <div key={message.id} className="flex items-start space-x-4">
//                     <div
//                       className={`w-8 h-8 rounded-full flex items-center justify-center ${
//                         message.type === "user" ? "bg-primary-blue" : "bg-card-bg border border-border-color"
//                       }`}
//                     >
//                       {message.type === "user" ? (
//                         <User className="text-white h-4 w-4" />
//                       ) : (
//                         <Bot className="text-primary-blue h-4 w-4" />
//                       )}
//                     </div>
//                     <div
//                       className={`flex-1 ${
//                         message.type === "user" ? "bg-primary-blue/10 p-4 rounded-lg" : "bg-transparent"
//                       }`}
//                     >
//                       <p className="whitespace-pre-wrap">{message.content}</p>
//                       <div className="text-xs text-text-secondary mt-2">
//                         {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
//                       </div>
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             )}
//           </div>

//           {feedback && (
//             <div className="p-4 border-t border-border-color bg-green-50">
//               <h3 className="font-medium mb-2">Interview Feedback</h3>
//               <p><strong>Score:</strong> {avgScore}</p>
//               <p>{feedback}</p>
//             </div>
//           )}
//         </div>
//       </div>
//     </>
//   );
// }


import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { MessageSquare, User, Bot, Send, Upload, FileText, Settings } from "lucide-react";
import FileUpload from "@/components/file-upload";
import Navigation from "@/components/navigation";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

interface InterviewChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  numQuestions: number;
  completed: boolean;
  avgScore?: number | null;
  messageCount: number;
}

interface PersistedSessionState {
  sessionId: string | null;
  messages: Message[];
  questions: string[];
  structuredQuestions: Array<any> | null;
  answers: string[];
  feedback: Array<{question: string; user_answer: string; score: number; feedback: string; correct_answer?: string}> | null;
  avgScore: number | null;
}

interface InterviewCopilotProps {
  isAuthenticated: boolean;
  userEmail: string;
  onLogout: () => void;
}

export default function InterviewCopilot({ 
  isAuthenticated, 
  userEmail, 
  onLogout 
}: InterviewCopilotProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [uploadedDoc, setUploadedDoc] = useState<File | null>(null);
  const [numQuestions, setNumQuestions] = useState<number>(5);
  const [questionType, setQuestionType] = useState<string>("technical");
  const [questions, setQuestions] = useState<string[]>([]);
  const [structuredQuestions, setStructuredQuestions] = useState<Array<any> | null>(null);
  const [answers, setAnswers] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  // feedback will be an array of per-question feedback objects from the server
  const [feedback, setFeedback] = useState<Array<{question: string; user_answer: string; score: number; feedback: string; correct_answer?: string}> | null>(null);
  const [avgScore, setAvgScore] = useState<number | null>(null);

  // Local chat history (persisted like Smart Documentation but via localStorage)
  const [chatSessions, setChatSessions] = useState<InterviewChatSession[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string>("");

  // Map frontend question types to backend levels
  const typeToLevelMap: Record<string, string> = {
    technical: "medium",
    mcq: "easy",
    hr: "easy",
    theory: "hard",
  };

  // ✅ Load user info and JWT token from localStorage
  const token = localStorage.getItem("token");
  // userEmail is supplied via props from App; no local sync needed here

  // ---------- Persistence Helpers ----------
  const SESSIONS_KEY = "interview_chat_sessions";
  const sessionStateKey = (id: string) => `interview_session_state_${id}`;

  const loadChatSessionsFromStorage = (): void => {
    try {
      const raw = localStorage.getItem(SESSIONS_KEY);
      if (!raw) return;
      const stored = JSON.parse(raw) as InterviewChatSession[];
      // revive dates
      const revived = stored.map(s => ({...s, timestamp: new Date(s.timestamp)}));
      setChatSessions(revived);
      if (revived.length > 0) {
        setCurrentChatId(revived[0].id);
        // load its state
        loadSessionState(revived[0].id);
      }
    } catch {}
  };

  const persistChatSessions = (sessions: InterviewChatSession[]): void => {
    try {
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
    } catch {}
  };

  const saveSessionState = (id: string): void => {
    if (!id) return;
    const snapshot: PersistedSessionState = {
      sessionId,
      messages,
      questions,
      structuredQuestions,
      answers,
      feedback,
      avgScore,
    };
    try {
      localStorage.setItem(sessionStateKey(id), JSON.stringify(snapshot));
    } catch {}
  };

  const loadSessionState = (id: string): void => {
    try {
      const raw = localStorage.getItem(sessionStateKey(id));
      if (!raw) return;
      const data = JSON.parse(raw) as PersistedSessionState;
      setSessionId(data.sessionId);
      // revive dates in messages
      const revivedMsgs = (data.messages || []).map(m => ({...m, timestamp: new Date(m.timestamp)}));
      setMessages(revivedMsgs);
      setQuestions(data.questions || []);
      setStructuredQuestions(data.structuredQuestions || null);
      setAnswers(data.answers || []);
      setFeedback(data.feedback || null);
      setAvgScore(data.avgScore ?? null);
    } catch {}
  };

  useEffect(() => {
    loadChatSessionsFromStorage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileUpload = (file: File) => {
    setUploadedDoc(file);
    setShowUpload(false);
    // Clear current session UI so previous chat isn't shown for the new document
    setMessages([]);
    setQuestions([]);
    setStructuredQuestions(null);
    setAnswers([]);
    setSessionId(null);
    setFeedback(null);
    setAvgScore(null);
    setCurrentChatId("");
  };

  // Start interview
  const startInterview = async () => {
    if (!uploadedDoc) {
      alert("Please upload a document.");
      return;
    }

    const formData = new FormData();
    formData.append("file", uploadedDoc);
  formData.append("num_questions", numQuestions.toString());
  formData.append("level", typeToLevelMap[questionType] || "medium");
  formData.append("question_type", questionType);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/interview/start", {
        method: "POST",
        body: formData,
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error ${response.status}: ${errorText}`);
      }

  const data = await response.json();

      if (!data.questions || data.questions.length === 0) {
        throw new Error("No questions generated");
      }

      // If the server returned structured questions (e.g., MCQ options), store them
      setStructuredQuestions(data.structured_questions ?? null);

      if (data.structured_questions && Array.isArray(data.structured_questions) && data.structured_questions.length > 0) {
        // use structured questions' text as display
        setQuestions(data.structured_questions.map((s: any) => s.question));
        setAnswers(new Array(data.structured_questions.length).fill(""));
      } else {
        setQuestions(data.questions);
        setAnswers(new Array(data.questions.length).fill(""));
      }
      setSessionId(data.session_id);

      // Display questions as bot messages
      const botMessage: Message = {
        id: Date.now().toString(),
        type: "bot",
        content: data.questions.map((q: string, idx: number) => `${idx + 1}. ${q}`).join("\n\n"),
        timestamp: new Date(),
      };
      setMessages((prev) => {
        const next = [...prev, botMessage];
        return next;
      });

      // Create a local chat session entry
      const newSession: InterviewChatSession = {
        id: sessionId || data.session_id,
        title: `Interview ${new Date().toLocaleString()}`,
        lastMessage: botMessage.content.slice(0, 100),
        timestamp: new Date(),
        numQuestions: (data.questions || []).length,
        completed: false,
        avgScore: null,
        messageCount: 1,
      };
      const updatedSessions = [newSession, ...chatSessions.filter(s => s.id !== newSession.id)];
      setChatSessions(updatedSessions);
      setCurrentChatId(newSession.id);
      persistChatSessions(updatedSessions);
      saveSessionState(newSession.id);
    } catch (error: any) {
      console.error("Error starting interview:", error);
      alert(error?.message || "Failed to start interview");
    }
  };

  // Update answer for a question
  const updateAnswer = (index: number, value: string) => {
    const updated = [...answers];
    updated[index] = value;
    setAnswers(updated);
  };

  // Submit all answers
  const submitAnswers = async () => {
    if (!sessionId) return;

    // Prevent re-submitting a completed session
    const current = chatSessions.find(s => s.id === (currentChatId || sessionId));
    if (feedback || (current && current.completed)) {
      return true;
    }

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/interview/${sessionId}/submit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ answers }), // ✅ backend expects 'answers'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error ${response.status}: ${errorText}`);
      }

      const data = await response.json();

      // The POST /submit returns { session_id, avg_score, feedback }
      const reviewPayload = {
        session_id: data.session_id,
        avg_score: data.avg_score ?? null,
        feedback: Array.isArray(data.feedback) ? data.feedback : [],
      };

      // Save to sessionStorage so the review page can render it after navigation
      try {
        sessionStorage.setItem("interview_review", JSON.stringify(reviewPayload));
      } catch (e) {
        console.warn("Failed to save review to sessionStorage", e);
      }

      setFeedback(reviewPayload.feedback);
      setAvgScore(reviewPayload.avg_score);

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "bot",
          content: `✅ Interview Completed!\nAverage Score: ${data.avg_score}`,
        timestamp: new Date(),
      };
      setMessages((prev) => {
        const next = [...prev, botMessage];
        return next;
      });

      // mark session completed and persist
      const updatedSessions = chatSessions.map(s =>
        s.id === (currentChatId || data.session_id)
          ? { ...s, completed: true, avgScore: reviewPayload.avg_score ?? null, lastMessage: `Average Score: ${reviewPayload.avg_score}`, timestamp: new Date(), messageCount: s.messageCount + 1 }
          : s
      );
      setChatSessions(updatedSessions);
      persistChatSessions(updatedSessions);
      saveSessionState(currentChatId || data.session_id);
    } catch (error: any) {
      console.error("Error submitting answers:", error);
      alert(error?.message || "Failed to submit answers");
      return false;
    }
    return true;
  };

  const [, setLocation] = useLocation();

  return (
    <>
      <Navigation 
        onAuthModal={() => {}} 
        isAuthenticated={isAuthenticated} 
        userEmail={userEmail}
        onLogout={onLogout}
      />
      <div className="flex h-screen bg-primary-bg pt-16">
        {/* Sidebar */}
        <div className="w-80 bg-card-bg border-r border-border-color flex flex-col">
          <div className="p-4 border-b border-border-color">
            <Button
              onClick={() => setShowUpload(!showUpload)}
              className="w-full bg-primary-blue hover:bg-blue-600 mb-3"
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload Document
            </Button>
          </div>

          {showUpload && (
            <div className="p-4 border-b border-border-color">
              <FileUpload
                onFileSelect={handleFileUpload}
                acceptedTypes=".pdf,.docx,.txt"
                maxSize={10}
                className="min-h-[100px]"
              />
            </div>
          )}

          <div className="p-4 border-b border-border-color">
            <h3 className="text-sm font-medium mb-3 flex items-center">
              <Settings className="h-4 w-4 mr-2" /> Question Settings
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-text-secondary">Number of Questions</label>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="text-xs text-text-secondary">Question Type</label>
                <Select onValueChange={setQuestionType} value={questionType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="technical">Technical</SelectItem>
                    <SelectItem value="mcq">MCQ</SelectItem>
                    <SelectItem value="hr">HR</SelectItem>
                    <SelectItem value="theory">Theory</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="p-4">
            {!sessionId ? (
              <Button onClick={startInterview} className="w-full bg-primary-blue hover:bg-blue-600">
                Start Interview
              </Button>
            ) : (
              <Button disabled className="w-full bg-gray-600 cursor-not-allowed">
                Questions Loaded
              </Button>
            )}
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-medium text-text-secondary mb-3">Recent Interviews</h3>
            {chatSessions.map((chat) => (
              <div
                key={chat.id}
                onClick={() => {
                  setCurrentChatId(chat.id);
                  loadSessionState(chat.id);
                }}
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
                  {chat.lastMessage || (chat.completed ? `Average Score: ${chat.avgScore ?? "-"}` : "No messages yet")}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Chat Section */}
        <div className="flex-1 flex flex-col">
          <div className="border-b border-border-color p-4 flex items-center justify-between">
            <MessageSquare className="text-primary-blue h-6 w-6 mr-3" />
            <h2 className="font-semibold">Interview Copilot</h2>
            <span className="text-sm text-text-secondary">Logged in as {userEmail}</span>
          </div>

          <div className="flex-1 p-6 overflow-y-auto">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-text-secondary">
                <FileText className="h-12 w-12 mb-3 text-primary-blue" />
                <p>Upload a document and I’ll generate interview questions for you.</p>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto space-y-6">
                {messages.map((message, i) => (
                  <div key={message.id} className="flex items-start space-x-4">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        message.type === "user" ? "bg-primary-blue" : "bg-card-bg border border-border-color"
                      }`}
                    >
                      {message.type === "user" ? (
                        <User className="text-white h-4 w-4" />
                      ) : (
                        <Bot className="text-primary-blue h-4 w-4" />
                      )}
                    </div>
                    <div
                      className={`flex-1 ${
                        message.type === "user" ? "bg-primary-blue/10 p-4 rounded-lg" : "bg-transparent"
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      <div className="text-xs text-text-secondary mt-2">
                        {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </div>
                    </div>
                  </div>
                ))}

                {/* Inline answer inputs */}
                {questions.map((q, idx) => (
                  <div key={idx} className="mt-4">
                    <p className="font-medium">{idx + 1}. {q}</p>
                    {/* If structured question has options (MCQ), render radio options */}
                    {structuredQuestions && structuredQuestions[idx] && Array.isArray(structuredQuestions[idx].options) && structuredQuestions[idx].options.length > 0 ? (
                      <div className="mt-2 space-y-2">
                        {structuredQuestions[idx].options.map((opt: string, oi: number) => (
                          <label key={oi} className="flex items-center space-x-3">
                            <input
                              type="radio"
                              name={`q-${idx}`}
                              value={opt}
                              checked={answers[idx] === opt}
                              onChange={() => {
                                updateAnswer(idx, opt);
                                const updatedSessions = chatSessions.map(s => s.id === currentChatId ? { ...s, lastMessage: `Answered Q${idx + 1}`, timestamp: new Date() } : s);
                                setChatSessions(updatedSessions);
                                persistChatSessions(updatedSessions);
                                saveSessionState(currentChatId);
                              }}
                            />
                            <span className="text-sm">{opt}</span>
                          </label>
                        ))}
                      </div>
                    ) : (
                      <Input
                        value={answers[idx] || ""}
                        onChange={(e) => {
                          updateAnswer(idx, e.target.value);
                          saveSessionState(currentChatId);
                        }}
                        placeholder="Type your answer here..."
                        className="mt-1"
                      />
                    )}
                  </div>
                ))}

                {/* Submit all answers button placed at the end of questions */}
                {questions.length > 0 && !feedback && (
                  <div className="mt-6">
                    <Button
                      onClick={async () => {
                        await submitAnswers();
                        // Navigate to review page
                        setLocation("/interview-review");
                      }}
                      className="w-full bg-green-600 hover:bg-green-700"
                    >
                      Submit All Answers & View Review
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>

          {feedback && (
            <div className="p-4 border-t border-border-color bg-green-50">
              <h3 className="font-medium mb-2">Interview Review</h3>
              <p className="mb-2"><strong>Average Score:</strong> {avgScore}</p>
              <div className="space-y-3">
                {feedback.map((fb, i) => (
                  <div key={i} className="p-3 bg-white rounded border border-border-color">
                    <div className="text-sm font-medium">{i + 1}. {fb.question}</div>
                    <div className="text-sm text-text-secondary mt-1"><strong>Your answer:</strong> {fb.user_answer}</div>
                    {fb.correct_answer && (
                      <div className="text-sm text-green-600 mt-1"><strong>Correct answer:</strong> {fb.correct_answer}</div>
                    )}
                    <div className="text-sm mt-1"><strong>Score:</strong> {fb.score}/10</div>
                    <div className="text-sm mt-1 text-text-secondary"><strong>Feedback:</strong> {fb.feedback}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
