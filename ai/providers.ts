import { createOpenAI } from "@ai-sdk/openai";
import { createGroq } from "@ai-sdk/groq";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createXai } from "@ai-sdk/xai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { ProviderOptions } from './types';

import { 
  customProvider, 
  wrapLanguageModel, 
  extractReasoningMiddleware 
} from "ai";

export interface ModelInfo {
  provider: string;
  name: string;
  description: string;
  apiVersion: string;
  capabilities: string[];
}

const middleware = extractReasoningMiddleware({
  tagName: 'think',
});

// Helper to get API keys from environment variables first, then localStorage
const getApiKey = (key: string): string | undefined => {
  // Check for environment variables first
  if (process.env[key]) {
    return process.env[key] || undefined;
  }
  
  // Fall back to localStorage if available
  if (typeof window !== 'undefined') {
    return window.localStorage.getItem(key) || undefined;
  }
  
  return undefined;
};

// Create provider instances with API keys from localStorage
const openaiClient = createOpenAI({
  apiKey: getApiKey('OPENAI_API_KEY'),
});

const anthropicClient = createAnthropic({
  apiKey: getApiKey('ANTHROPIC_API_KEY'),
});

const groqClient = createGroq({
  apiKey: getApiKey('GROQ_API_KEY'),
});

const xaiClient = createXai({
  apiKey: getApiKey('XAI_API_KEY'),
});

const googleClient = createGoogleGenerativeAI({
  apiKey: getApiKey('GOOGLE_GENERATIVE_AI_API_KEY'),
  baseURL: 'https://generativelanguage.googleapis.com/v1beta'
});

const languageModels = {
  "gpt-4.1-mini": openaiClient("gpt-4.1-mini"),
  "claude-3-7-sonnet": anthropicClient('claude-3-7-sonnet-20250219'),
  "qwen-qwq": wrapLanguageModel(
    {
      model: groqClient("qwen-qwq-32b"),
      middleware
    }
  ),
  "grok-3-mini": xaiClient("grok-3-mini-latest"),
  "gemini-pro": wrapLanguageModel(
    {
      model: googleClient("gemini-pro"),
      middleware,
      providerOptions: {
        google: {
          responseModalities: ['TEXT'],
          thinkingConfig: {
            thinkingBudget: 1024
          },
          safetySettings: [
            { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' }
          ]
        }
      }
    }
  ),
  "gemini-2-5-flash-preview": wrapLanguageModel(
    {
      model: googleClient("gemini-2.5-flash-preview-04-17"),
      middleware,
      providerOptions: {
        google: {
          responseModalities: ['TEXT'],
          thinkingConfig: {
            thinkingBudget: 1024
          },
          safetySettings: [
            { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' }
          ]
        }
      }
    }
  ),
  "gemini-2-5-pro-exp": wrapLanguageModel(
    {
      model: googleClient("gemini-2.5-pro-exp-03-25"),
      middleware,
      providerOptions: {
        google: {
          responseModalities: ['TEXT'],
          thinkingConfig: {
            thinkingBudget: 1024
          },
          safetySettings: [
            { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' }
          ]
        }
      }
    }
  ),
  "gemini-2-0-flash": wrapLanguageModel(
    {
      model: googleClient("gemini-2.0-flash"),
      middleware,
      providerOptions: {
        google: {
          responseModalities: ['TEXT'],
          thinkingConfig: {
            thinkingBudget: 1024
          },
          safetySettings: [
            { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
            { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' }
          ]
        }
      }
    }
  ),
};

export const modelDetails: Record<keyof typeof languageModels, ModelInfo> = {
  "gpt-4.1-mini": {
    provider: "OpenAI",
    name: "GPT-4.1 Mini",
    description: "Compact version of OpenAI's GPT-4.1 with good balance of capabilities, including vision.",
    apiVersion: "gpt-4.1-mini",
    capabilities: ["Balance", "Creative", "Vision"]
  },
  "claude-3-7-sonnet": {
    provider: "Anthropic",
    name: "Claude 3.7 Sonnet",
    description: "Latest version of Anthropic's Claude 3.7 Sonnet with strong reasoning and coding capabilities.",
    apiVersion: "claude-3-7-sonnet-20250219",
    capabilities: ["Reasoning", "Efficient", "Agentic"]
  },
  "qwen-qwq": {
    provider: "Groq",
    name: "Qwen QWQ",
    description: "Latest version of Alibaba's Qwen QWQ with strong reasoning and coding capabilities.",
    apiVersion: "qwen-qwq",
    capabilities: ["Reasoning", "Efficient", "Agentic"]
  },
  "grok-3-mini": {
    provider: "XAI",
    name: "Grok 3 Mini",
    description: "Latest version of XAI's Grok 3 Mini with strong reasoning and coding capabilities.",
    apiVersion: "grok-3-mini-latest",
    capabilities: ["Reasoning", "Efficient", "Agentic"]
  },
  "gemini-pro": {
    provider: "Google",
    name: "Gemini Pro",
    description: "Google's Gemini Pro model with strong reasoning, coding, and multimodal capabilities.",
    apiVersion: "gemini-pro",
    capabilities: ["Reasoning", "Multimodal", "Coding"]
  },
  "gemini-2-5-flash-preview": {
    provider: "Google",
    name: "Gemini 2.5 Flash Preview",
    description: "Latest preview version of Gemini 2.5 Flash with improved speed and capabilities.",
    apiVersion: "gemini-2.5-flash-preview-04-17",
    capabilities: ["Fast", "Preview", "Improved"]
  },
  "gemini-2-5-pro-exp": {
    provider: "Google",
    name: "Gemini 2.5 Pro Experimental",
    description: "Experimental version of Gemini 2.5 Pro with advanced features and capabilities.",
    apiVersion: "gemini-2.5-pro-exp-03-25",
    capabilities: ["Advanced", "Experimental", "Enhanced"]
  },
  "gemini-2-0-flash": {
    provider: "Google",
    name: "Gemini 2.0 Flash",
    description: "Fast and efficient version of Gemini 2.0 optimized for quick responses.",
    apiVersion: "gemini-2.0-flash",
    capabilities: ["Fast", "Efficient", "Optimized"]
  },
};

// Update API keys when localStorage changes (for runtime updates)
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    // Reload the page if any API key changed to refresh the providers
    if (event.key?.includes('API_KEY')) {
      window.location.reload();
    }
  });
}

export const model = customProvider({
  languageModels,
});

export type modelID = keyof typeof languageModels;

export const MODELS = Object.keys(languageModels);

export const defaultModel: modelID = "qwen-qwq";
