using System;
using System.Threading.Tasks;
using Frontend;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama; // Provides AddOllamaChatCompletion extension
using Microsoft.SemanticKernel.Connectors.OpenAI; // Provides AddOpenAIChatCompletion extension
using Microsoft.SemanticKernel.Connectors.HuggingFace; // Provides AddHuggingFaceChatCompletion extension
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol.Transport;
using ModelContextProtocol.Protocol.Types;
using DotNetEnv; // Added for .env file support

#pragma warning disable SKEXP0070 // Acknowledge experimental status of Ollama connector
#pragma warning disable SKEXP0110 // Acknowledge experimental status of Ollama connector
#pragma warning disable SKEXP0071 // Acknowledge experimental status of HuggingFace connector

public class Program
{
    public static async Task Main(string[] args)
    {
        // Clear the console for a fresh start
        Console.Clear();
        
        // Load environment variables from .env file
        DotNetEnv.Env.Load("/Users/toto/Projects/JetsonNano/ai-video-solution/Frontend/.env");

        // Stylish console banner
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔═══════════════════════════════════════════════════╗");
        Console.WriteLine("║                                                   ║");
        Console.WriteLine("║       🤖  AI Vision Assistant Console  🎥        ║");
        Console.WriteLine("║                                                   ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine();

        // Determine if running in online mode (using OpenAI) or Hugging Face
        bool useOpenAI = args.Contains("-online");
        bool useHuggingFace = args.Contains("-huggingface");

        // Configure these to your Ollama setup
        var ollamaMode_text_lId = "gpt-oss:20b"; // Or your preferred model, e.g., "mistral", "phi3"
        //var ollamaMode_text_lId = "llama3.2"; // Or your preferred model, e.g., "mistral", "phi3"
        var ollamaMode_vision_lId = "minicpm-v:8b"; // Changed from :latest to :8b (the actual installed version)
        //var ollamaMode_vision_lId = "llava-phi3:latest"; // Or your preferred model, e.g., "mistral", "phi3"
        var ollamaBaseUrl = new Uri("http://localhost:11434"); // Default Ollama API endpoint

        // Configure these for your OpenAI setup
        var openAIModelId = "o4-mini"; // E.g., "gpt-4"
        // Read API key from environment variable
        var openAIApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

        // Configure these for your Hugging Face setup
        var huggingFaceModelId = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"; // Or your preferred model
        var huggingFaceApiKey = Environment.GetEnvironmentVariable("HUGGINGFACE_API_KEY");

        if (useOpenAI && string.IsNullOrEmpty(openAIApiKey))
        {
            Console.WriteLine("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable in your .env file.");
            // Potentially fallback or exit, for now, we let the logic below handle it
        }

        if (useHuggingFace && string.IsNullOrEmpty(huggingFaceApiKey))
        {
            Console.WriteLine("Hugging Face API key is not set. Please set the HUGGINGFACE_API_KEY environment variable in your .env file.");
            // Potentially fallback or exit
        }

        var builder = Kernel.CreateBuilder();

        // Add Ollama Chat Completion using the recommended extension method
        builder.AddOllamaChatCompletion(
            modelId: ollamaMode_text_lId,
            endpoint: ollamaBaseUrl, // Pass the Uri object directly
            serviceId: "ollamaTxt"      // Specify a service key for keyed retrieval
        );

        builder.AddOllamaChatCompletion(
            modelId: ollamaMode_vision_lId,
            endpoint: ollamaBaseUrl, // Pass the Uri object directly
            serviceId: "ollamaVis"      // Specify a service key for keyed retrieval
        );

        // Add OpenAI Chat Completion only if using OpenAI and API key is present
        if (useOpenAI && !string.IsNullOrEmpty(openAIApiKey))
        {
            builder.AddOpenAIChatCompletion(
                modelId: openAIModelId,
                apiKey: openAIApiKey, // Use the key read from .env
                serviceId: "openAI" // Specify a service key for keyed retrieval
            );
        }

        // Add Hugging Face Chat Completion if using HuggingFace and API key is present
        if (useHuggingFace /*&& !string.IsNullOrEmpty(huggingFaceApiKey)*/)
        {
            builder.AddHuggingFaceChatCompletion(
                model: huggingFaceModelId,
                apiKey: huggingFaceApiKey,
                serviceId: "huggingFace" // Specify a service key for keyed retrieval
            );
        }

        var cwd = Path.GetDirectoryName(Environment.ProcessPath);
        Console.WriteLine($"Current working directory: {cwd}");
        // Create an MCPClient for the GitHub server
        await using IMcpClient mcpClient = await McpClientFactory.CreateAsync(new StdioClientTransport(new()
        {
            Name = "Video",
            Command = "uv",
            Arguments = ["run",
            "--with", "mcp", "mcp", "run", "/Users/toto/Projects/JetsonNano/ai-video-solution/mcp_server/mcp_server.py"],
        }));

        var tools = await mcpClient.ListToolsAsync().ConfigureAwait(false);

        var kernel = builder.Build(); // Build the kernel AFTER all services have been added
        //var kernelTxt = kernel.Clone(); // Cloning will be conditional or handled differently
        //var kernelVis = kernel.Clone();

        var settingsTxt = new OllamaPromptExecutionSettings { 
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(autoInvoke: true),
            ServiceId = "ollamaTxt",
            Temperature = 0.3f  // Increased from 0.1 for better reasoning flexibility
        };
        var settingsVis = new OllamaPromptExecutionSettings { 
            FunctionChoiceBehavior = FunctionChoiceBehavior.None(), 
            ServiceId = "ollamaVis"
        };
        var settingsOpenAI = new OpenAIPromptExecutionSettings { 
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(autoInvoke: true),
            ServiceId = "openAI",
            Temperature = 0.3f,  // Increased from 0.1 for better reasoning
            MaxTokens = 1500     // Increased from 500 for more complete responses
        };
        var settingsHuggingFace = new HuggingFacePromptExecutionSettings { 
            ServiceId = "huggingFace"
        }; // FunctionChoiceBehavior not available

        // Retrieve the chat completion service
        var chatCompletionServiceOllamaTxt = kernel.Services.GetRequiredKeyedService<IChatCompletionService>("ollamaTxt");
        var chatCompletionServiceOllamaVis = kernel.Services.GetRequiredKeyedService<IChatCompletionService>("ollamaVis");
        IChatCompletionService? chatCompletionServiceOpenAI = null; // Initialize as nullable
        if (useOpenAI && !string.IsNullOrEmpty(openAIApiKey)) // Only retrieve if it should have been added
        {
            chatCompletionServiceOpenAI = kernel.Services.GetKeyedService<IChatCompletionService>("openAI");
        }

        IChatCompletionService? chatCompletionServiceHuggingFace = null; // Initialize as nullable
        if (useHuggingFace /*&& !string.IsNullOrEmpty(huggingFaceApiKey)*/) // Only retrieve if it should have been added
        {
            chatCompletionServiceHuggingFace = kernel.Services.GetKeyedService<IChatCompletionService>("huggingFace");
        }


        // Create the plugin once, as it's used by both OpenAI and Ollama (Txt)
        var pluginsObjectDetection = KernelPluginFactory.CreateFromObject(new Tools(chatCompletionServiceOllamaVis), "objectDetection");

        IChatCompletionService activeChatService;
        PromptExecutionSettings? activeSettings;
        Kernel activeKernel; 

        if (useOpenAI && chatCompletionServiceOpenAI != null) // Check if OpenAI service was successfully retrieved
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Using OpenAI API with function calling.");
            Console.ResetColor();
            activeChatService = chatCompletionServiceOpenAI;
            activeSettings = settingsOpenAI;
            kernel.Plugins.Add(pluginsObjectDetection); // Add plugins to the main kernel for OpenAI
            activeKernel = kernel; 
        }
        else if (useHuggingFace && chatCompletionServiceHuggingFace != null) // Check if Hugging Face service was successfully retrieved
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Using Hugging Face API.");
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Using Ollama API (either no specific service requested, API key missing, or service retrieval failed) with function calling.");
            Console.ResetColor();
            activeChatService = chatCompletionServiceOllamaTxt; // Default to Ollama text
            activeSettings = settingsTxt;
            var kernelTxt = kernel.Clone(); // Clone the kernel for Ollama-specific plugins
            kernelTxt.Plugins.Add(pluginsObjectDetection); // Add plugins to the cloned kernel for Ollama Txt
            activeKernel = kernelTxt; 
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Using Ollama API (either no specific service requested, API key missing, or service retrieval failed) with function calling.");
            Console.ResetColor();
            activeChatService = chatCompletionServiceOllamaTxt; // Default to Ollama text
            activeSettings = settingsTxt;
            var kernelTxt = kernel.Clone(); // Clone the kernel for Ollama-specific plugins
            kernelTxt.Plugins.Add(pluginsObjectDetection); // Add plugins to the cloned kernel for Ollama Txt
            activeKernel = kernelTxt; 
        }

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine("═══════════════════════════════════════════════════");
        Console.WriteLine($"  Chat Session Started - Type 'exit' to quit");
        Console.WriteLine("═══════════════════════════════════════════════════");
        Console.ResetColor();
        Console.WriteLine();
        
        // Add function invocation logging
        activeKernel.FunctionInvocationFilters.Add(new FunctionInvocationLoggingFilter());
        
        var chatHistory = new ChatHistory("""
        You are a Vision Assistant for a VIDEO OBJECT DETECTION SYSTEM.

        CRITICAL CONTEXT:
        - You work with OBJECT DETECTION data (people, chairs, cups, bottles, etc.)
        - Each detected object has a TRACK_ID (an integer like 2, 3, 5, 31)
        - You help users identify and track specific objects in video frames
        - THIS IS NOT MUSIC DATA - ignore any audio/music interpretations

        WHEN USER ASKS "what ids are there?" or "list objects":
        1. Call GetCurrentDetectionsAsync()
        2. The response contains object detection data with:
           - "label": type of object (person, chair, cup, etc.)
           - "track_id": unique integer ID for tracking
           - "box": bounding box coordinates
        3. Parse the "objects" field and list each item clearly
        4. Example response: "I see: person (ID: 2), person (ID: 5), chair (ID: 3), cup (ID: 7)"

        TRACKING WORKFLOW:
        1. User asks to track an object
        2. Call GetTheImage() to see the video frame
        3. Identify the object's ID from visual label
        4. Call SetTrackIdFilterAsync(ID) with the integer ID
        5. Confirm tracking is active

        GOLDEN RULES:
        - NEVER interpret detection data as music/audio data
        - ALWAYS parse JSON responses correctly
        - track_id = object tracking ID (NOT music metadata)
        - label = object type (NOT record label)
        - box = bounding box coordinates (NOT audio frames)
        - Stay focused on OBJECT DETECTION domain
        """);

        const int MAX_HISTORY_MESSAGES = 10; // Keep system message + last 10 exchanges (increased from 3)
        
        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("You ▶ ");
            Console.ResetColor();
            var userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput))
            {
                continue;
            }

            if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("╔═══════════════════════════════════════════════════╗");
                Console.WriteLine("║          👋  Thank you for using the AI           ║");
                Console.WriteLine("║              Vision Assistant Console             ║");
                Console.WriteLine("╚═══════════════════════════════════════════════════╝");
                Console.ResetColor();
                break;
            }

            chatHistory.AddUserMessage(userInput);

            try
            {
                // Trim chat history to prevent context overflow
                // Keep system message (index 0) and last N messages
                if (chatHistory.Count > MAX_HISTORY_MESSAGES + 1) // +1 for system message
                {
                    int messagesToRemove = chatHistory.Count - MAX_HISTORY_MESSAGES - 1;
                    for (int i = 0; i < messagesToRemove; i++)
                    {
                        chatHistory.RemoveAt(1); // Always remove at index 1 (after system message)
                    }
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"   [Context trimmed: Keeping last {MAX_HISTORY_MESSAGES} messages]");
                    Console.ResetColor();
                }

                // Use GetChatMessageContentsAsync to enable automatic function calling loop
                // This will keep calling functions until the LLM provides a final answer
                var results = await activeChatService.GetChatMessageContentsAsync(
                    chatHistory,
                    activeSettings,
                    kernel: activeKernel
                ).ConfigureAwait(false);

                // Get the final response (last message in the results)
                var finalResult = results.LastOrDefault();
                var assistantResponse = finalResult?.Content ?? "No response generated.";

                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("AI 🤖 ");
                Console.ResetColor();
                Console.WriteLine(assistantResponse);
                
                // Add visual separator after each conversation
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine("───────────────────────────────────────────────────");
                Console.ResetColor();
                Console.WriteLine();
                
                chatHistory.AddAssistantMessage(assistantResponse);
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.ForegroundColor = ConsoleColor.DarkRed;
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                Console.ResetColor();
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine("───────────────────────────────────────────────────");
                Console.ResetColor();
                Console.WriteLine();
                
                // Optionally, remove the last user message if the API call failed to allow retry or different input
                if (chatHistory.Count > 0)
                {
                    chatHistory.RemoveAt(chatHistory.Count - 1);
                }
            }
        }
    }


    private static KernelFunction ConfigureSelectionFunction(string? nameVisual, string? nameDetection) =>
            AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
               **Task:** Determine which agent should act next based on the last response. Respond with **only** the agent's name from the list below. Do not add any explanations or extra text.

               **Agents:**

               - {{{nameVisual}}} 
               - {{{nameDetection}}}

               **Selection Rules:**

               - **If the last response is from the user:** Choose **{{{nameVisual}}} **.
               - **If the last response is from {{{nameVisual}}}  and it is requesting an action:** Choose **{{{nameDetection}}}**.
               - **If the last response is from {{{nameDetection}}} :** Choose **User**.
               - **Never select the same agent who provided the last response.**

               **Last Response:**

               {{$lastmessage}}
               """,
                safeParameterNames: "lastmessage"
            );

    private static KernelFunction ConfigureTerminationFunction() =>
        AgentGroupChat.CreatePromptFunctionForStrategy(
            $$$"""
               **Task:** Determine, if the 'agentDetection' approved or rejected.

               **Rules:**
               - you can end this conversation, if 'agentDetection' approved.
               - you can only approve, if last message was from 'agentDetection'.
               - **If the agentDetection has approved:** Respond with **YES**.
               - **If there are still unresolved issues or action is required:** Respond with **NO**.

               **Communication**:
               Provide a simple and short explanation about your decision.
               Provide a last single line with only **YES** or **NO**. Do not add any additional text to that line.

               **Last Response:**

               {{$lastmessage}}
               """,
            safeParameterNames: "lastmessage"
        );

    private static ChatCompletionAgent ConfigureAgentDetection(string nameDetection, Kernel kernel, ILoggerFactory loggerFactory) =>
        new()
        {
            Name = nameDetection,
            Instructions =
                """
                **Role:** You are an experienced object analyst specializing in visual analysis.
                 You have extensive knowledge of image processing, object detection, visual description.

                **Task:**

                - Analyze the request from the user.
                - You have to get current detections of objects in a video stream.
                - You have to get the current systems state.
                - You have to get snapshots of the video stream.

                **Guidelines:**

                - Be methodical and systematic in your approach.
                - Communicate clearly and specifically when asking for information.
                - Do not provide solutions; focus only on analysis.
                - Keep it simple.

                **Communication Style:**

                - Use simple and clear language.
                - Avoid unnecessary technical jargon.
                - Be professional and collaborative.
                - Keep your answer always short and simple.

                """,
            Kernel = kernel,
            LoggerFactory = loggerFactory
        };

    private static ChatCompletionAgent ConfigureAgentIntend(string nameIntend, Kernel kernel, ILoggerFactory loggerFactory) =>
        new()
        {
            Name = nameIntend,
            Instructions =
                """
                **Role:** You are an experienced Orchestrator and manager.
                 You have extensive experience in understanding requirements and intentions.
                 You are able to identify the intent of the user and decide which agent should act next.
                 Your main decision fields are object detection, visual analysis.

                **Task:**

                - Analyze the request from the user.
                - Identify the intent of the user.
                - Decide if the request is asking for a specific object or a general analysis.
                - If the user wants to take action on a specific object, call the agent for detection.
                - If the user is asking for a general analysis, provide a detailed description of the visual data.
                - Any intent that needs more details in an image should be passed to the agent for visual analysis.
                - Any intent that is related to object detection should be passed to the agent for detection.
                
                **Guidelines:**

                - Be methodical and systematic in your approach.
                - Communicate clearly and specifically when asking for information.
                - Do not provide solutions; focus only on analysis.
                - Keep it simple.

                **Communication Style:**

                - Use simple and clear language.
                - Avoid unnecessary technical jargon.
                - Be professional and collaborative.
                - Keep your answer always short and simple.

                """,
            Kernel = kernel,
            LoggerFactory = loggerFactory
        };

    private static ChatCompletionAgent ConfigureAgentVisual(string nameVisual, Kernel kernel, ILoggerFactory loggerFactory) =>
        new()
        {
            Name = nameVisual,
            Instructions =
                """
                **Role:** You are an experienced IT professional specializing in visual analysis.
                 You have extensive knowledge of image processing, object detection, visual description.

                **Task:**

                - Analyze the request from the user.
                - Identitfy the intend of the user.
                - Provide a detailed analysis of the visual data specifc to the user's request.
                - If the user is asking for a specific object, provide a detailed description of the object.
                

                **Guidelines:**

                - Be methodical and systematic in your approach.
                - Communicate clearly and specifically when asking for information.
                - Do not provide solutions; focus only on analysis.
                - Keep it simple.

                **Communication Style:**

                - Use simple and clear language.
                - Avoid unnecessary technical jargon.
                - Be professional and collaborative.
                - Keep your answer always short and simple.

                """,
            Kernel = kernel,
            LoggerFactory = loggerFactory
        };
}

// Function invocation logging filter to track tool calls
public class FunctionInvocationLoggingFilter : IFunctionInvocationFilter
{
    public async Task OnFunctionInvocationAsync(FunctionInvocationContext context, Func<FunctionInvocationContext, Task> next)
    {
        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"   🔧 Calling tool: {context.Function.Name}");
        
        // Log parameters if any
        if (context.Arguments.Count > 0)
        {
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine($"      Parameters: {string.Join(", ", context.Arguments.Select(kvp => $"{kvp.Key}={kvp.Value}"))}");
        }
        
        Console.ResetColor();
        
        await next(context);
        
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"   ✅ Tool {context.Function.Name} completed");
        Console.ResetColor();
        Console.WriteLine();
    }
}

#pragma warning restore SKEXP0070
#pragma warning restore SKEXP0110
#pragma warning restore SKEXP0071