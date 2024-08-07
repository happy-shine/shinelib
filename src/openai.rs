/// 
/// Created by happy-shine on 2024-08-07.
/// 
/// 对 Openai 标准 API 调用的封装.
/// 
/// 
/// TODO:
///   - 还需完善参数列表
///   - File, Image, Function Calling 
///   - Retry 机制
/// 


use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use futures::{Stream, StreamExt};
use serde_json::from_str;

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    pub max_tokens: i32,
    pub temperature: f32,
}

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
}

#[derive(Debug, Deserialize)]
pub struct CompletionChunk {
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
}


#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
    pub encoding_format: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}


pub struct OpenAI {
    pub api_key: String,
    pub base_url: String,
}

impl OpenAI {
    pub fn new(api_key: String, base_url: Option<String>) -> OpenAI {
        let mut base_url = base_url.unwrap_or_else(|| "https://api.openai.com/v1/".to_string());
    
        if !base_url.ends_with('/') {
            base_url.push('/');
        }

        OpenAI {
            api_key,
            base_url
        }
    }

    pub async fn completions(
        &self, 
        model: String, 
        messages: Vec<Message>,
        max_tokens: i32,
        temperature: f32
    ) -> Result<String, Box<dyn std::error::Error>> {
        let client = Client::new();
        
        let request_body = CompletionRequest {
            model: model,
            messages,
            stream: false,
            max_tokens,
            temperature
        };

        let response = client.post(format!("{}chat/completions", self.base_url))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await?;

        let completion: CompletionResponse = response.json().await?;

        if let Some(choice) = completion.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err("No completion choices returned".into())
        }
    }

    pub fn stream_completions(
        &self,
        model: String,
        messages: Vec<Message>,
        max_tokens: i32,
        temperature: f32
    ) -> impl Stream<Item = Result<String, Box<dyn Error + Send + Sync>>> + Send + 'static {
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();

        async_stream::try_stream! {
            let client = Client::new();
            
            let request_body = CompletionRequest {
                model,
                messages,
                stream: true,
                max_tokens,
                temperature
            };

            let response = client.post(format!("{}chat/completions", base_url))
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&request_body)
                .send()
                .await?;

            let mut stream = response.bytes_stream();

            let mut buffer = Vec::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                buffer.extend_from_slice(&chunk);

                while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                    let line = String::from_utf8_lossy(&buffer[..pos]).to_string();
                    buffer.drain(..=pos);

                    if line.starts_with("data: ") {
                        let data = line.trim_start_matches("data: ");
                        if data == "[DONE]" {
                            return;
                        } else {
                            match from_str::<CompletionChunk>(data) {
                                Ok(chunk) => {
                                    if let Some(choice) = chunk.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            yield content.clone();
                                        }
                                    }
                                }
                                Err(e) => yield Err(Box::new(e) as Box<dyn Error + Send + Sync>)?,
                            }
                        }
                    }
                }
            }
        }
    }

    pub async fn embeddings(&self, input: Vec<String>, model: String) -> Result<EmbeddingResponse, Box<dyn std::error::Error>> {
        let client = Client::new();
        
        let request_body = EmbeddingRequest {
            input,
            model,
            encoding_format: "float".to_string(),
        };

        let response = client.post(format!("{}embeddings", self.base_url))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await?;

        let embedding_response: EmbeddingResponse = response.json().await?;

        Ok(embedding_response)
    }
}

