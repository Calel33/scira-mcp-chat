// Types for Google Generative AI provider options

export type HarmCategory = 
  | 'HARM_CATEGORY_DANGEROUS_CONTENT'
  | 'HARM_CATEGORY_HATE_SPEECH'
  | 'HARM_CATEGORY_HARASSMENT'
  | 'HARM_CATEGORY_SEXUALLY_EXPLICIT';

export type HarmBlockThreshold = 
  | 'BLOCK_LOW_AND_ABOVE'
  | 'BLOCK_MEDIUM_AND_ABOVE'
  | 'BLOCK_HIGH_AND_ABOVE'
  | 'BLOCK_ONLY_HIGH';

export interface SafetySetting {
  category: HarmCategory;
  threshold: HarmBlockThreshold;
}

export interface ThinkingConfig {
  thinkingBudget: number;
}

export type ResponseModality = 'TEXT' | 'IMAGE';

export interface GoogleProviderOptions {
  responseModalities: ResponseModality[];
  thinkingConfig: ThinkingConfig;
  safetySettings: SafetySetting[];
}

export interface ProviderOptions {
  google: GoogleProviderOptions;
} 