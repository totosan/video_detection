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
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol.Transport;
using ModelContextProtocol.Protocol.Types;

#pragma warning disable SKEXP0070 // Acknowledge experimental status of Ollama connector
#pragma warning disable SKEXP0110 // Acknowledge experimental status of Ollama connector

public class Program
{
    public static async Task Main(string[] args)
    {
        // Configure these to your Ollama setup
        var ollamaMode_text_lId = "llama3.2"; // Or your preferred model, e.g., "mistral", "phi3"
        var ollamaMode_vision_lId = "gemma3:4b-it-qat"; // Or your preferred model, e.g., "mistral", "phi3"
        var ollamaBaseUrl = new Uri("http://localhost:11434"); // Default Ollama API endpoint

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

        var kernel = builder.Build();
        var kernelTxt = kernel.Clone();
        //var kernelVis = kernel.Clone();

        var settingsTxt = new OllamaPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(), ServiceId = "ollamaTxt" };
        var settingsVis = new OllamaPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.None(), ServiceId = "ollamaVis" };


        // Retrieve the chat completion service
        var chatCompletionService = kernel.Services.GetRequiredKeyedService<IChatCompletionService>("ollamaTxt");
        var chatCompletionServiceVis = kernel.Services.GetRequiredKeyedService<IChatCompletionService>("ollamaVis");

        var pluginsObjectDetection = KernelPluginFactory.CreateFromObject(new Tools(chatCompletionServiceVis),"objectDetection");
        kernelTxt.Plugins.Add(pluginsObjectDetection);        
        

        Console.WriteLine("Chat with Ollama model (type 'exit' to quit):");
        var chatHistory = new ChatHistory(@"You are a helpful AI assistant. Answer the user's questions clearly and concisely.");

        while (true)
        {
            Console.Write("User: ");
            var userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput))
            {
                continue;
            }

            if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Exiting chat.");
                break;
            }

            chatHistory.AddUserMessage(userInput);

            try
            {
                var result = await chatCompletionService.GetChatMessageContentAsync(chatHistory, settingsTxt, kernel: kernelTxt).ConfigureAwait(false);
                var assistantResponse = result.Content;

                Console.WriteLine($"Assistant: {assistantResponse}");
                chatHistory.AddAssistantMessage(assistantResponse ?? string.Empty);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                // Optionally, remove the last user message if the API call failed to allow retry or different input
                // chatHistory.Messages.RemoveAt(chatHistory.Messages.Count - 1);
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

#pragma warning restore SKEXP0070
#pragma warning restore SKEXP0110