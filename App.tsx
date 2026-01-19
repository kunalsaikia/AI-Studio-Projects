
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { Turn } from './types';
import { createPcmBlob, decode, decodeAudioData } from './utils/audio';

const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-12-2025';

type AudioSource = 'microphone' | 'system';

const App: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [history, setHistory] = useState<Turn[]>([]);
  const [currentInterviewerText, setCurrentInterviewerText] = useState('');
  const [currentAiText, setCurrentAiText] = useState('');
  const [activeSegments, setActiveSegments] = useState<string[]>([]);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resume, setResume] = useState<string>(localStorage.getItem('interview_resume') || '');
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioSource, setAudioSource] = useState<AudioSource>('microphone');
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);

  // Refs for audio processing
  const audioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const sessionRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const isListeningRef = useRef(false);

  // Synchronize ref with state for the audio processor closure
  useEffect(() => {
    isListeningRef.current = isListening;
  }, [isListening]);

  // Persistence
  useEffect(() => {
    localStorage.setItem('interview_resume', resume);
  }, [resume]);

  // Auto-scroll logic
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentInterviewerText, currentAiText, history]);

  const stopSession = useCallback(() => {
    setIsActive(false);
    setIsListening(false);
    window.speechSynthesis.cancel();
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    sourcesRef.current.forEach(source => {
      try { source.stop(); } catch (e) {}
    });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;
  }, []);

  const getSystemInstruction = () => {
    let base = `
You are an expert real-time technical interview co-pilot. 
You will receive live transcriptions of an interviewer's questions.
Your goal is to provide immediate, high-impact suggested answers for the candidate to use.

RULES:
- For behavioral questions: Use a concise bulleted STAR method (Situation, Task, Action, Result) summary.
- For technical questions: Provide clear concepts, pseudo-code, or exact code snippets.
- Use simple, direct, conversational language that is easy to read out loud.
- STRICTLY NO FILLER. Do not say "Here is a response" or "I can help with that."
- PERSONALIZATION: Incorporate specific projects, skills, and experiences from the provided resume context.

CITATIONS:
At the very end of your response, after a double newline, provide the exact snippets from the 'CANDIDATE RESUME CONTEXT' that you actually used to generate your answer. 
Format it exactly like this:
RESUME_USAGE: ["snippet 1", "snippet 2"]
`;
    if (resume.trim()) {
      base += `\n\nCANDIDATE RESUME CONTEXT:\n${resume}\n`;
    }
    return base;
  };

  const startSession = async () => {
    try {
      setError(null);
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      audioContextRef.current = inputCtx;
      outputAudioContextRef.current = outputCtx;

      let stream: MediaStream;
      if (audioSource === 'system') {
        try {
          stream = await navigator.mediaDevices.getDisplayMedia({
            video: { displaySurface: 'browser' },
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true }
          } as any);
          
          const audioTracks = stream.getAudioTracks();
          if (audioTracks.length === 0) {
            stream.getTracks().forEach(t => t.stop());
            throw new Error('SYSTEM_AUDIO_MISSING');
          }
        } catch (e: any) {
          if (e.name === 'NotAllowedError') {
            throw new Error('Permission denied. Selection required for "Meet" mode.');
          }
          throw new Error(e.message || 'Capture failed.');
        }
      } else {
        stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          } 
        });
      }
      
      streamRef.current = stream;

      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: getSystemInstruction(),
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
        },
        callbacks: {
          onopen: () => {
            setIsActive(true);
            const source = inputCtx.createMediaStreamSource(stream);
            const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              if (sessionRef.current && isListeningRef.current) {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmBlob = createPcmBlob(inputData);
                sessionRef.current.sendRealtimeInput({ media: pcmBlob });
              }
            };
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputCtx.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.inputTranscription) {
              setCurrentInterviewerText(prev => prev + message.serverContent!.inputTranscription!.text);
            }
            
            if (message.serverContent?.outputTranscription) {
              setIsGenerating(true);
              setCurrentAiText(prev => prev + message.serverContent!.outputTranscription!.text);
            }

            if (message.serverContent?.turnComplete) {
              setIsGenerating(false);
              setIsListening(false); // Automatically stop listening when a turn is complete
              
              const rawText = currentAiText;
              let cleanAnswer = rawText;
              let snippets: string[] = [];

              const citationMatch = rawText.match(/RESUME_USAGE:\s*(\[.*\])/s);
              if (citationMatch) {
                try {
                  snippets = JSON.parse(citationMatch[1]);
                  cleanAnswer = rawText.replace(/RESUME_USAGE:\s*\[.*\]/s, '').trim();
                } catch (e) {
                  console.error("Failed to parse citations", e);
                }
              }

              if (cleanAnswer) {
                const turnId = Date.now().toString();
                setHistory(prev => [
                  ...prev,
                  {
                    id: turnId,
                    interviewer: currentInterviewerText,
                    aiSuggested: cleanAnswer,
                    timestamp: Date.now(),
                    usedSegments: snippets
                  },
                ]);
                setSelectedTurnId(turnId);
                setActiveSegments(snippets);
                setCurrentInterviewerText('');
                setCurrentAiText('');
              }
            }

            const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (audioData && outputCtx && isVoiceEnabled) {
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
              const audioBuffer = await decodeAudioData(decode(audioData), outputCtx, 24000, 1);
              const source = outputCtx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputCtx.destination);
              source.addEventListener('ended', () => {
                sourcesRef.current.delete(source);
              });
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => { try { s.stop(); } catch(e) {} });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setIsGenerating(false);
              setCurrentAiText('');
              setActiveSegments([]);
            }
          },
          onerror: (e) => {
            setError('Connection error. Check your API key or network.');
            stopSession();
          },
          onclose: () => setIsActive(false),
        },
      });

      sessionRef.current = await sessionPromise;
    } catch (err: any) {
      if (err.message === 'SYSTEM_AUDIO_MISSING') {
        setError('IMPORTANT: You must check the "Share tab audio" checkbox in the popup to use Meet mode.');
      } else {
        setError(err.message || 'Capture failed.');
      }
      setIsActive(false);
    }
  };

  const toggleListening = () => {
    if (!isActive) return;
    setIsListening(!isListening);
  };

  const handleGenerateAnswer = () => {
    if (sessionRef.current && currentInterviewerText) {
      setIsListening(false);
      setIsGenerating(true);
      sessionRef.current.send({
        parts: [{ text: `Answer this interviewer question based on my resume: "${currentInterviewerText}"` }]
      });
    }
  };

  const handleNextQuestion = () => {
    setCurrentInterviewerText('');
    setCurrentAiText('');
    setIsGenerating(false);
    setIsListening(false);
    setActiveSegments([]);
    setSelectedTurnId(null);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        setResume(text);
      };
      reader.readAsText(file);
    }
  };

  const handleSpeakText = (text: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  const selectHistoryItem = (turn: Turn) => {
    setSelectedTurnId(turn.id);
    setActiveSegments(turn.usedSegments || []);
  };

  const highlightedResume = useMemo(() => {
    if (!resume) return null;
    if (activeSegments.length === 0) return resume;

    const sortedSegments = [...activeSegments].sort((a, b) => b.length - a.length);
    const escapedSegments = sortedSegments.map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const combinedRegex = new RegExp(`(${escapedSegments.join('|')})`, 'gi');

    const parts = resume.split(combinedRegex);
    return parts.map((part, i) => {
      const isMatch = sortedSegments.some(s => s.toLowerCase() === part.toLowerCase());
      if (isMatch) {
        return (
          <span 
            key={i} 
            className="bg-indigo-500/30 text-indigo-100 rounded px-0.5 shadow-[0_0_8px_rgba(99,102,241,0.3)] transition-all animate-pulse"
          >
            {part}
          </span>
        );
      }
      return part;
    });
  }, [resume, activeSegments]);

  const displayAiText = useMemo(() => {
    return currentAiText.replace(/RESUME_USAGE:\s*\[.*\]/s, '').trim();
  }, [currentAiText]);

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-500 z-50"></div>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,#1e1b4b,transparent_40%)] opacity-30 pointer-events-none"></div>

      <header className="relative z-10 flex items-center justify-between px-8 py-4 border-b border-slate-800 bg-slate-900/60 backdrop-blur-2xl">
        <div className="flex items-center space-x-4">
          <div className={`w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 ${isActive ? 'animate-pulse' : ''}`}>
             <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight flex items-center">
              INTERVIEW <span className="text-indigo-500 ml-1.5 underline decoration-2 decoration-indigo-500/30 underline-offset-4">COPILOT</span>
            </h1>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="bg-slate-800/80 rounded-xl p-1 border border-slate-700 flex mr-2">
            <button 
              onClick={() => { if(!isActive) setAudioSource('microphone'); }}
              className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${isActive ? 'cursor-not-allowed opacity-50' : 'hover:bg-slate-700/50'} ${audioSource === 'microphone' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500'}`}
            >
              Mic Only
            </button>
            <button 
              onClick={() => { if(!isActive) setAudioSource('system'); }}
              className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${isActive ? 'cursor-not-allowed opacity-50' : 'hover:bg-slate-700/50'} ${audioSource === 'system' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500'}`}
            >
              Meet/Tab
            </button>
          </div>

          <button
            onClick={isActive ? stopSession : startSession}
            className={`px-8 py-2.5 rounded-xl font-black text-sm transition-all duration-300 transform active:scale-95 shadow-2xl ${
              isActive 
                ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/40' 
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/40'
            }`}
          >
            {isActive ? 'STOP' : 'CONNECT'}
          </button>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden relative z-10">
        
        {/* LEFT COLUMN: CONTROLS */}
        <aside className="w-72 lg:w-80 flex flex-col border-r border-slate-800 bg-slate-900/20 overflow-hidden">
          <div className="p-4 border-b border-slate-800 bg-slate-900/40">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Controls</h3>
          </div>
          <div className="p-6 space-y-6">
            <button 
              onClick={() => setIsVoiceEnabled(!isVoiceEnabled)}
              className={`w-full py-4 rounded-xl border text-[10px] font-black uppercase tracking-widest transition-all flex items-center justify-center space-x-2 ${isVoiceEnabled ? 'bg-indigo-500/10 border-indigo-500/50 text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.1)]' : 'bg-slate-800/80 border-slate-700 text-slate-500 hover:text-slate-300'}`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>
              <span>{isVoiceEnabled ? 'AI Voice Enabled' : 'AI Voice Muted'}</span>
            </button>
            <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-4">
              <h4 className="text-[9px] font-black text-slate-500 uppercase mb-3">Session Stats</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-black/20 p-3 rounded-lg border border-slate-800/50">
                  <div className="text-[8px] text-slate-500 uppercase">Questions</div>
                  <div className="text-xl font-bold text-white">{history.length}</div>
                </div>
                <div className="bg-black/20 p-3 rounded-lg border border-slate-800/50">
                  <div className="text-[8px] text-slate-500 uppercase">Res. Coverage</div>
                  <div className="text-xl font-bold text-indigo-400">
                    {history.length > 0 ? Math.round((history.filter(h => (h.usedSegments?.length || 0) > 0).length / history.length) * 100) : 0}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* CENTER COLUMN: LIVE FEED & HISTORY FLOW */}
        <section className="flex-1 flex flex-col overflow-hidden bg-slate-950/40 backdrop-blur-sm relative">
          <div className="flex-1 flex flex-col p-8 overflow-y-auto no-scrollbar max-w-4xl mx-auto w-full space-y-12">
            
            {/* 1. Live Transcript Area */}
            <div className="flex flex-col space-y-6">
               <div className="flex items-center justify-between">
                 <div className="flex items-center space-x-3">
                   <div className={`w-2 h-2 rounded-full ${isListening ? 'bg-emerald-500 animate-pulse' : 'bg-slate-700'}`}></div>
                   <h2 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                     {isListening ? 'Listening to Interviewer...' : 'Mic Paused (Safe to Talk)'}
                   </h2>
                 </div>
               </div>
               <div className={`text-2xl md:text-3xl font-bold leading-tight italic min-h-[80px] transition-all duration-300 ${isListening ? 'text-slate-100' : 'text-slate-600'}`}>
                  {currentInterviewerText || (isActive ? (isListening ? "Transcribing question..." : "Ready to listen...") : "Connect to start...")}
               </div>
               
               <div className="flex items-center space-x-4">
                  {/* Primary Listening Button */}
                  <button 
                    disabled={!isActive || isGenerating}
                    onClick={toggleListening}
                    className={`px-8 py-4 rounded-2xl text-sm font-black uppercase tracking-widest transition-all shadow-xl flex items-center space-x-3 transform active:scale-95 ${
                      isListening 
                        ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20' 
                        : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/20'
                    }`}
                  >
                    {isListening ? (
                       <svg className="w-5 h-5 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" /></svg>
                    ) : (
                       <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                    )}
                    <span>{isListening ? "I'm Done Listening" : "Listen to Interviewer"}</span>
                  </button>

                  <button 
                    disabled={!currentInterviewerText || isGenerating || !isActive}
                    onClick={handleGenerateAnswer}
                    className="px-6 py-4 bg-slate-800 hover:bg-slate-700 disabled:opacity-20 text-slate-100 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all border border-slate-700"
                  >
                    {isGenerating ? "Analyzing..." : "Analyze & Answer"}
                  </button>

                  <button 
                    disabled={!currentInterviewerText && !displayAiText || isGenerating}
                    onClick={handleNextQuestion}
                    className="px-4 py-4 bg-slate-900 hover:bg-slate-800 disabled:opacity-20 text-slate-500 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all"
                  >
                    Next
                  </button>
               </div>
            </div>

            {/* 2. Active Suggested Answer */}
            {displayAiText && (
              <div className="animate-in slide-in-from-bottom-8 duration-500">
                <div className="flex items-center space-x-3 mb-6">
                  <div className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]"></div>
                  <h2 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Active Suggestion</h2>
                </div>
                <div className="bg-indigo-600/10 border border-indigo-500/30 p-10 rounded-[2.5rem] shadow-2xl relative group ring-4 ring-indigo-500/5">
                   <div className="absolute -top-4 -right-4 w-12 h-12 bg-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-900/40">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9l-.707.707M12 18c-4.418 0-8-3.582-8-8s3.582-8 8-8 8 3.582 8 8-3.582 8-8 8z" /></svg>
                   </div>
                   <button 
                      onClick={() => handleSpeakText(displayAiText)}
                      className="absolute bottom-4 right-4 p-2 rounded-xl bg-indigo-500/20 text-indigo-400 opacity-0 group-hover:opacity-100 transition-all hover:bg-indigo-500 hover:text-white"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>
                    </button>
                   <div className="text-xl md:text-2xl font-medium leading-relaxed text-indigo-50 whitespace-pre-wrap">
                     {displayAiText}
                   </div>
                </div>
              </div>
            )}

            {/* 3. PRIOR NOTES */}
            {history.length > 0 && (
              <div className="pt-12 border-t border-slate-800 space-y-8">
                <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest text-center">Previous Turns</h3>
                <div className="flex flex-col-reverse space-y-8 space-y-reverse">
                  {history.map((turn) => (
                    <div 
                      key={turn.id} 
                      onClick={() => selectHistoryItem(turn)}
                      className={`cursor-pointer bg-slate-900/30 border rounded-3xl p-6 transition-all group ${selectedTurnId === turn.id ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-800 hover:border-slate-700'}`}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <span className="text-[8px] font-bold text-slate-600 uppercase tracking-widest">{new Date(turn.timestamp).toLocaleTimeString()}</span>
                        <div className="flex items-center space-x-2">
                           <button onClick={(e) => { e.stopPropagation(); handleSpeakText(turn.aiSuggested); }} className="text-slate-500 hover:text-indigo-400 p-1">
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" /></svg>
                          </button>
                        </div>
                      </div>
                      <p className="text-xs text-slate-500 italic mb-4 line-clamp-1 group-hover:line-clamp-none transition-all">"{turn.interviewer}"</p>
                      <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
                        {turn.aiSuggested}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div ref={transcriptEndRef} />
          </div>
        </section>

        {/* RIGHT COLUMN: RESUME */}
        <aside className="w-72 lg:w-80 flex flex-col border-l border-slate-800 bg-slate-900/20 overflow-hidden">
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="p-6 pb-2 flex items-center justify-between">
              <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Resume Context</h3>
              <label className="cursor-pointer text-[10px] font-black uppercase text-indigo-500 hover:text-indigo-400">
                Update
                <input type="file" accept=".txt,.md" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>
            <div className="flex-1 p-6 overflow-hidden">
              <div className="h-full bg-slate-950/50 border border-slate-800 rounded-2xl p-4 overflow-y-auto no-scrollbar font-mono text-[10px] text-slate-500 leading-relaxed group">
                {resume ? (
                  <div className="whitespace-pre-wrap">
                    {highlightedResume}
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center p-4 space-y-4">
                     <svg className="w-8 h-8 text-slate-800" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                     <p className="font-bold">No resume uploaded. Suggested answers will be generic.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </aside>

      </main>

      {/* Global Error Notifications */}
      {error && (
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-[100] w-full max-w-lg px-6">
           <div className="bg-rose-600 text-white p-6 rounded-[2rem] shadow-[0_20px_50px_rgba(225,29,72,0.4)] flex flex-col space-y-4 font-bold border border-rose-400/30">
             <div className="flex items-start space-x-4">
               <svg className="w-8 h-8 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
               <div className="flex-1 space-y-2">
                 <p className="text-base leading-tight">{error}</p>
                 {error.includes('tab audio') && (
                   <div className="text-[10px] bg-black/20 p-3 rounded-xl leading-normal font-medium">
                     Switch to **Mic Only** mode to bypass this dialog entirely.
                   </div>
                 )}
               </div>
               <button onClick={() => setError(null)} className="hover:opacity-50 transition-opacity"><svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
             </div>
           </div>
        </div>
      )}

      {/* Footer Instructions */}
      <footer className="h-10 border-t border-slate-800 bg-slate-900 px-8 flex items-center justify-between text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <span className="text-slate-600">Active Path:</span>
            <span className={audioSource === 'microphone' ? 'text-emerald-500' : 'text-indigo-400'}>{audioSource === 'microphone' ? 'Mic (No Dialog)' : 'System Capture'}</span>
          </div>
          <span className="text-slate-800">|</span>
          <div className="flex items-center space-x-2">
            <span className="text-slate-600">Privacy:</span>
            <span className={isListening ? 'text-rose-500' : 'text-emerald-500'}>{isListening ? 'LISTENING ON' : 'MIC PAUSED (SAFE)'}</span>
          </div>
        </div>
        <div className="hidden sm:block">
           Click 'Listen to Interviewer' only when they are asking a question.
        </div>
      </footer>
    </div>
  );
};

export default App;
