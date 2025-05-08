using System;
using System.ComponentModel;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using System.Collections.Generic;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;

#pragma warning disable SKEXP0070 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
namespace Frontend
{
    /// <summary>
    /// A class that provides Plugin methods for Semantic Kernel to access VideoStream Object Detector APIs.
    /// </summary>
    public class Tools
    {
        private readonly HttpClient client = new HttpClient();
        private const string FlaskApiBaseUrl = "http://localhost:3000/api";
        private const string FlaskAppBaseUrl = "http://localhost:3000"; // For snapshot endpoints

        private readonly IChatCompletionService _chatCompletionService;

        public Tools(IChatCompletionService completionService)
        {
            _chatCompletionService = completionService;
        }

        private async Task<string> GetApiResponseAsync(string url)
        {
            HttpResponseMessage response = await client.GetAsync(url);
            response.EnsureSuccessStatusCode(); // Throw for bad status codes
            return await response.Content.ReadAsStringAsync();
        }

        private async Task<string> PostApiResponseAsync(string url, HttpContent? content)
        {
            HttpResponseMessage response = await client.PostAsync(url, content);
            response.EnsureSuccessStatusCode(); // Throw for bad status codes
            return await response.Content.ReadAsStringAsync();
        }

        [KernelFunction, Description(@"Checks the running status of the detection system. Returns JSON: {""is_running"": true/false} or {""is_running"": false, ""error"": ""..."" }")]
        public async Task<string> GetSystemStatusAsync()
        {
            string url = $"{FlaskApiBaseUrl}/status";
            try
            {
                // The python /api/status returns {"running": status}, so we adjust the key name here to match mcp_server.py tool description
                string apiResponse = await GetApiResponseAsync(url);
                var jsonDoc = JsonDocument.Parse(apiResponse);
                if (jsonDoc.RootElement.TryGetProperty("running", out JsonElement runningElement))
                {
                    return JsonSerializer.Serialize(new { is_running = runningElement.GetBoolean() });
                }
                return JsonSerializer.Serialize(new { is_running = false, error = "Unexpected API response format from /api/status" });
            }
            catch (HttpRequestException e)
            {
                return JsonSerializer.Serialize(new { is_running = false, error = $"API request failed: {e.Message}" });
            }
            catch (Exception e)
            {
                return JsonSerializer.Serialize(new { is_running = false, error = $"Unexpected error: {e.Message}" });
            }
        }

        [KernelFunction, Description(@"Retrieves the latest raw detection results (bounding boxes, confidences, classes). Returns JSON: {""detections"": [...]} or {""error"": ""...""}")]
        public async Task<string> GetCurrentDetectionsAsync()
        {
            string url = $"{FlaskApiBaseUrl}/current_detections";
            try
            {
                return await GetApiResponseAsync(url); // app.py returns {"detections": [...]} or {"error": "..."}
            }
            catch (HttpRequestException e)
            {
                return JsonSerializer.Serialize(new { detections = Array.Empty<object>(), error = $"API request failed: {e.Message}" });
            }
            catch (Exception e)
            {
                return JsonSerializer.Serialize(new { detections = Array.Empty<object>(), error = $"Unexpected error: {e.Message}" });
            }
        }



        [KernelFunction, Description(@"Toggles the generation of annotated video frames on the backend server. Returns JSON: {""backend_annotation_enabled"": true/false} or {""error"": ""...""}")]
        public async Task<string> ToggleBackendAnnotationAsync()
        {
            string url = $"{FlaskApiBaseUrl}/backend_annotation/toggle";
            try
            {
                // POST request with empty content
                return await PostApiResponseAsync(url, null); // app.py returns {"backend_annotation_enabled": status}
            }
            catch (HttpRequestException e)
            {
                return JsonSerializer.Serialize(new { error = $"API request failed: {e.Message}" });
            }
            catch (Exception e)
            {
                return JsonSerializer.Serialize(new { error = $"Unexpected error: {e.Message}" });
            }
        }

        [KernelFunction, Description(@"Requests a snapshot of the latest processed frame as a base64 encoded string. Returns JSON: {""status"": ""success"", ""image_base64"": ""...""} or {""status"": ""error"", ""message"": ""...""}")]
        public async Task<string> GetSnapshotAsync()
        {

            string url = $"{FlaskAppBaseUrl}/snapshot"; // Uses the app base URL
            try
            {
                HttpResponseMessage response = await client.GetAsync(url);
                response.EnsureSuccessStatusCode();
                byte[] imageBytes = await response.Content.ReadAsByteArrayAsync();
                string base64Image = Convert.ToBase64String(imageBytes);
                // build the cahtcompletionmessage for a vision model
                var userMessage = new ChatMessageContentItemCollection
                {
                  new TextContent("""
                  Task: as a vision model, please describe the image in detail.
                  
                  Rules:
                  - Provide a detailed description of the image.
                  - Include information about the objects, their positions, and any relevant context.
                  - Use clear and concise language.
                  - keep it simple
                  
                  Format:
                    - Use JSON format for the response.
                    - Include the following keys in the JSON response:
                        - objects: List of objects detected in the image.
                           - object[0]: { "name": "object_name", "position": <Descripttion in prosa> }

                  """),
                    new ImageContent(new ReadOnlyMemory<byte>(Convert.FromBase64String(base64Image)), "image/jpeg")
                };
                var @chat = new ChatHistory();
                chat.AddUserMessage(userMessage);
                var result = await _chatCompletionService.GetChatMessageContentAsync(chat);

                return result.Content;
            }
            catch (HttpRequestException e)
            {
                return JsonSerializer.Serialize(new { status = "error", message = $"API request failed: {e.Message}" });
            }
            catch (Exception e)
            {
                return JsonSerializer.Serialize(new { status = "error", message = $"Unexpected error: {e.Message}" });
            }
        }

        // [KernelFunction, Description(@"Requests a raw snapshot directly from the RTSP stream as a base64 encoded string. Returns JSON: {""status"": ""success"", ""image_base64"": ""...""} or {""status"": ""error"", ""message"": ""...""}")]
        // public async Task<string> GetRawSnapshotAsync()
        // {
        //     string url = $"{FlaskAppBaseUrl}/raw_snapshot"; // Uses the app base URL
        //     try
        //     {
        //         HttpResponseMessage response = await client.GetAsync(url);
        //         response.EnsureSuccessStatusCode();
        //         byte[] imageBytes = await response.Content.ReadAsByteArrayAsync();
        //         string base64Image = Convert.ToBase64String(imageBytes);
        //         return JsonSerializer.Serialize(new { status = "success", image_base64 = base64Image });
        //     }
        //     catch (HttpRequestException e)
        //     {
        //         return JsonSerializer.Serialize(new { status = "error", message = $"API request failed: {e.Message}" });
        //     }
        //     catch (Exception e)
        //     {
        //         return JsonSerializer.Serialize(new { status = "error", message = $"Unexpected error: {e.Message}" });
        //     }
        // }
    }
}

#pragma warning restore SKEXP0070 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
