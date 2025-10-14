using System;
using System.ComponentModel;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using System.Collections.Generic;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using Microsoft.VisualBasic;

#pragma warning disable SKEXP0070 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
namespace Frontend
{
    /// <summary>
    /// A class that provides Plugin methods for Semantic Kernel to access VideoStream Object Detector APIs.
    /// </summary>
    public class Tools
    {
        private readonly HttpClient client;
        private const string FlaskApiBaseUrl = "http://localhost:8082/api";
        private const string FlaskAppBaseUrl = "http://localhost:8082"; // For snapshot endpoints

        private readonly IChatCompletionService _chatCompletionService;

        public Tools(IChatCompletionService completionService)
        {
            _chatCompletionService = completionService;
            
            // Initialize HttpClient with User-Agent header
            client = new HttpClient();
            client.DefaultRequestHeaders.Add("User-Agent", "AI-Vision-Assistant/1.0");
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

        [KernelFunction, Description(@"Requests the current objects, that one can see in a scene. Returns JSON: {""detections"": [...]} or {""error"": ""...""}. This always returns ALL objects in the scene, regardless of any active tracking filters, so you can help users select objects to track.")]
        public async Task<string> GetCurrentDetectionsAsync()
        {
            // Use the unfiltered endpoint so we can see ALL objects even when a track ID filter is active
            string url = $"{FlaskApiBaseUrl}/current_detections_unfiltered";
            try
            {
                var result = await GetApiResponseAsync(url); // app.py returns {"detections": [...]} or {"error": "..."}
                return result;
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

        [KernelFunction, Description(@"Describes the image from the video camera. Use this tool when the user asks 'What can you see?' or 'describe, what you see'. It provides a detailed analysis of the image, including objects and their positions.")]
        public async Task<string> GetTheImage()
        {
            Console.WriteLine("Getting current snapshot");
            string snapshotUrl = $"{FlaskAppBaseUrl}/snapshot"; // Uses the app base URL
            string detectionsUrl = $"{FlaskApiBaseUrl}/current_detections_unfiltered"; // Get ALL detection metadata (unfiltered)
            
            try
            {
                // Fetch both the image and the detection metadata in parallel
                var snapshotTask = client.GetAsync(snapshotUrl);
                var detectionsTask = client.GetAsync(detectionsUrl);
                
                await Task.WhenAll(snapshotTask, detectionsTask);
                
                var snapshotResponse = await snapshotTask;
                var detectionsResponse = await detectionsTask;
                
                Console.WriteLine($"Requesting snapshot from: {snapshotUrl}");
                Console.WriteLine($"Response status: {snapshotResponse.StatusCode}");
                
                if (!snapshotResponse.IsSuccessStatusCode)
                {
                    string errorContent = await snapshotResponse.Content.ReadAsStringAsync();
                    Console.WriteLine($"Error response: {errorContent}");
                    return JsonSerializer.Serialize(new { 
                        status = "error", 
                        message = $"Failed to get snapshot. Status: {snapshotResponse.StatusCode}, Details: {errorContent}" 
                    });
                }
                
                snapshotResponse.EnsureSuccessStatusCode();
                byte[] imageBytes = await snapshotResponse.Content.ReadAsByteArrayAsync();
                Console.WriteLine($"Received {imageBytes.Length} bytes");
                
                // Get detection data
                string detectionsJson = "{}";
                if (detectionsResponse.IsSuccessStatusCode)
                {
                    detectionsJson = await detectionsResponse.Content.ReadAsStringAsync();
                    Console.WriteLine($"Retrieved detection metadata: {detectionsJson.Length} characters");
                }
                else
                {
                    Console.WriteLine($"Warning: Failed to get detections (status: {detectionsResponse.StatusCode})");
                }
                
                using var image = Image.Load(imageBytes);
                int width = image.Width;
                int height = image.Height;
                Console.WriteLine($"Image size: {width}x{height}");
                
                double scale = 0.75; // reduce size
                int newWidth = (int)(width * scale);
                int newHeight = (int)(height * scale); 

                image.Mutate(ctx => ctx.Resize(newWidth, newHeight));
                using var msResized = new MemoryStream();
                image.SaveAsJpeg(msResized);
                var base64Image = Convert.ToBase64String(msResized.ToArray());
               

                // Save resized image to disk
                var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "Snapshots");
                Directory.CreateDirectory(outputDir);
                var fileName = $"snapshot_{DateTime.UtcNow:yyyyMMdd_HHmmss}.jpg";
                var filePath = Path.Combine(outputDir, fileName);
                await File.WriteAllBytesAsync(filePath, msResized.ToArray());
                Console.WriteLine($"Saved snapshot to: {filePath}");

                // build the chatcompletionmessage for a vision model
                var userMessage = new ChatMessageContentItemCollection
                {
                  new TextContent($"""
                  You are analyzing an annotated image from an object detection system. The image has bounding boxes drawn on it.
                  
                  GROUND TRUTH DETECTION DATA:
                  {detectionsJson}
                  
                  INSTRUCTIONS:
                  1. Use the detection data above as the PRIMARY source of truth for object identification
                  2. Each detection includes: label (object type), track_id (unique ID), bbox (bounding box coordinates), and confidence
                  3. Describe the scene based on these ACTUAL detections, supplemented by visual context from the image
                  4. Report the EXACT track_ids from the detection data - do NOT make up IDs
                  
                  Provide a JSON response with:
                  1. "description": Brief scene summary (1 sentence)
                  2. "objects": Array of detected objects from the detection data
                  
                  For each object provide:
                  - "label": Object type from detection data
                  - "track_id": The exact track_id from the detection data
                  - "position_description": Spatial location based on bbox coordinates:
                    * Horizontal: "left" (<33%), "center" (33-66%), "right" (>66%)
                    * Vertical: "top" (<33%), "middle" (33-66%), "bottom" (>66%)
                    * Depth: Estimate from size/overlap: "foreground", "midground", "background"
                  - "confidence": Confidence score from detection data
                  - "notable_features": Visual characteristics you can see (optional)
                  
                  CRITICAL: Use ONLY the track_ids and labels from the detection data provided above.
                  """),
                    new ImageContent(new ReadOnlyMemory<byte>(Convert.FromBase64String(base64Image)), "image/jpeg")
                };
                var @chat = new ChatHistory();
                chat.AddUserMessage(userMessage);
                var result = await _chatCompletionService.GetChatMessageContentAsync(chat);

                if (result?.Content is string content)
                {
                    return content;
                }

                return JsonSerializer.Serialize(new { status = "error", message = "Failed to get a description from the vision model." });
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine($"HTTP request error: {e.Message}");
                Console.WriteLine($"Stack trace: {e.StackTrace}");
                return JsonSerializer.Serialize(new { status = "error", message = $"API request failed: {e.Message}" });
            }
            catch (Exception e)
            {
                Console.WriteLine($"Unexpected error: {e.Message}");
                Console.WriteLine($"Stack trace: {e.StackTrace}");
                return JsonSerializer.Serialize(new { status = "error", message = $"Unexpected error: {e.Message}" });
            }
        }

        [KernelFunction, Description(@"Highlight or select objects. Use as parameter the name of the object. More objects are separated by a comma.")]
        public async Task<string> SetObjectFilterAsync(string objectLabels)
        {
            string url = $"{FlaskApiBaseUrl}/set_object_filter";
            try
            {
                var labels = objectLabels
                    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                var payload = new { object_filter = labels };
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    System.Text.Encoding.UTF8,
                    "application/json");
                return await PostApiResponseAsync(url, content); // app.py handles the filter setting
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


        [KernelFunction, Description(@" Getting the active object filter from the backend server. Returns JSON: {""filter"": {...}} or {""error"": ""...""}")]
        public async Task<string> GetObjectFilterAsync()
        {
            string url = $"{FlaskApiBaseUrl}/get_object_filter";
            try
            {
                return await GetApiResponseAsync(url); // app.py returns the current filter
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

        [KernelFunction, Description(@"Track a specific object by its track ID. Use this when you have identified a specific object (by matching vision analysis with current detections) and want to follow only that object. The track_id must be obtained from GetCurrentDetectionsAsync. Setting this filter will clear any active label filter (filters are mutually exclusive). Returns JSON: {""message"": ""..."", ""track_id_filter"": <id>} or {""error"": ""...""}")]
        public async Task<string> SetTrackIdFilterAsync(int trackId)
        {
            string url = $"{FlaskApiBaseUrl}/set_track_id_filter";
            try
            {
                var payload = new { track_id = trackId };
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    System.Text.Encoding.UTF8,
                    "application/json");
                
                string response = await PostApiResponseAsync(url, content);
                return response; // Backend returns {"message": "...", "track_id_filter": trackId}
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

        [KernelFunction, Description(@"Get the currently active track ID filter. Returns the track_id of the object being tracked, or null if no track ID filter is active. Use this when the user asks 'what are you tracking?' or 'what object is being followed?'. Returns JSON: {""track_id_filter"": <id>} or {""track_id_filter"": null} or {""error"": ""...""}")]
        public async Task<string> GetTrackIdFilterAsync()
        {
            string url = $"{FlaskApiBaseUrl}/get_track_id_filter";
            try
            {
                string response = await GetApiResponseAsync(url);
                return response; // Backend returns {"track_id_filter": <id or null>}
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

        [KernelFunction, Description(@"Clear all active filters (both object label filters and track ID filters). Use this when the user wants to see all objects again, reset the filtering state, or says 'show me everything' or 'clear filter'. Returns JSON: {""message"": ""..."", ""object_filter"": []} or {""error"": ""...""}")]
        public async Task<string> ClearFiltersAsync()
        {
            string url = $"{FlaskApiBaseUrl}/set_object_filter";
            try
            {
                // Clearing by setting object_filter to empty array also clears track_id filter
                var payload = new { object_filter = new string[] { } };
                var content = new StringContent(
                    JsonSerializer.Serialize(payload),
                    System.Text.Encoding.UTF8,
                    "application/json");
                
                string response = await PostApiResponseAsync(url, content);
                return response; // Backend returns {"message": "...", "object_filter": []}
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
    }
}

#pragma warning restore SKEXP0070 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
