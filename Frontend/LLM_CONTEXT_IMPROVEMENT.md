# LLM Context Degradation - Root Causes & Solutions

## Problem Summary
After a few conversations with Llama and MiniCPM, the LLM output becomes "unsharp" and stops respecting prompted rules. **Additionally, the model hallucinates completely incorrect interpretations of tool results (e.g., interpreting object detection JSON as music streaming data).**

## Root Causes Identified

### 1. **Hallucination of Tool Results** ⚠️ CRITICAL ISSUE
- **Symptom**: Model interprets `{"detections": [...], "label": "chair", "track_id": 3}` as music streaming data
- **Root Cause**: Llama 3.2 struggles with raw JSON interpretation without explicit guidance
- **Impact**: Completely incorrect responses despite correct tool calls
- **Fix Applied**: 
  - Enhanced tool descriptions with explicit field meanings
  - Pre-formatted tool results into human-readable format
  - Added domain-specific context to system prompt ("THIS IS NOT MUSIC DATA")

### 2. **Aggressive Context Trimming** ⚠️ CRITICAL
- **Previous Setting**: Only keeping 3 messages + system prompt
- **Impact**: Model loses conversation context after 2-3 exchanges
- **Fix Applied**: Increased to 10 messages
- **Why This Matters**: LLMs need sufficient context to maintain coherent behavior

### 2. **Overly Restrictive Temperature** 
- **Previous Setting**: 0.1 for both Ollama and OpenAI
- **Impact**: Extremely deterministic, can lead to "stuck" behavior and poor reasoning
- **Fix Applied**: Increased to 0.3
- **Why This Matters**: Temperature 0.1-0.2 is too restrictive for multi-turn conversations

### 3. **Token Limit Too Low (OpenAI)**
- **Previous Setting**: MaxTokens = 500
- **Impact**: Responses get cut off, incomplete reasoning
- **Fix Applied**: Increased to 1500
- **Why This Matters**: Vision + reasoning tasks need more tokens

### 4. **Weak System Prompt Structure**
- **Previous**: Brief, informal instructions
- **Fix Applied**: Structured, emphatic prompt with clear rules and workflows
- **Why This Matters**: Local models need explicit, repeated reminders

## Changes Made

### ✅ 0. Fixed Tool Result Hallucinations (MOST CRITICAL)
**Problem**: Model was interpreting detection data as music metadata
**Solution**: 
```csharp
// Enhanced tool description
[KernelFunction, Description(@"Gets all detected objects currently visible in the video frame. 
Returns a list of objects with their labels (e.g., 'person', 'chair', 'cup') and track IDs (integers). 
Each detection has: label (object type), track_id (unique integer ID), box (coordinates), and color.")]

// Pre-format JSON for better understanding
return JsonSerializer.Serialize(new { 
    summary = $"Found {detections.Count} objects",
    objects = detections, // ["person (ID: 2)", "chair (ID: 3)"]
    raw_data = result 
});
```

**System Prompt Enhancement**:
```
CRITICAL CONTEXT:
- You work with OBJECT DETECTION data (people, chairs, cups, bottles, etc.)
- THIS IS NOT MUSIC DATA - ignore any audio/music interpretations
- track_id = object tracking ID (NOT music metadata)
- label = object type (NOT record label)
```

### ✅ 1. Increased Context Window
```csharp
const int MAX_HISTORY_MESSAGES = 10; // Was 3
```

### ✅ 2. Better Temperature Settings
```csharp
Temperature = 0.3f  // Was 0.1f
```

### ✅ 3. Increased Token Budget
```csharp
MaxTokens = 1500  // Was 500 (OpenAI only)
```

### ✅ 4. Enhanced System Prompt
- Added "CORE RULES (ALWAYS FOLLOW)" section
- Emphasized "STRICT ORDER" for workflows
- Added "CRITICAL REMINDERS" section
- More explicit language throughout

## Additional Recommendations

### For Production Use:

#### 1. **Context Window Monitoring**
Add token counting to track when you're approaching model limits:
```csharp
// Add before GetChatMessageContentsAsync
int estimatedTokens = EstimateTokenCount(chatHistory);
if (estimatedTokens > 6000) // Adjust based on model
{
    Console.WriteLine($"⚠️  High token usage: {estimatedTokens}. Consider resetting.");
}
```

#### 2. **Smart Context Pruning**
Instead of removing oldest messages, keep:
- System prompt (always)
- Last user question (always)
- Critical messages containing tool results
- Recent N messages

#### 3. **Model-Specific Settings**
```csharp
// Llama 3.2 - good balance
Temperature = 0.3f
// MiniCPM-V (vision) - slightly higher for creativity
Temperature = 0.4f
```

#### 4. **Context Reset Command**
Add a way for users to reset context:
```csharp
if (userInput.Equals("reset", StringComparison.OrdinalIgnoreCase))
{
    chatHistory.Clear();
    chatHistory.AddSystemMessage(originalSystemPrompt);
    Console.WriteLine("✓ Context reset.");
    continue;
}
```

#### 5. **Vision Model Considerations**
MiniCPM-V has special requirements:
- Images consume significant context tokens
- Consider clearing old images from history
- Keep only last 2-3 images in context

```csharp
// Periodically remove old images from context
if (chatHistory.Count > 15)
{
    // Remove messages containing images, keep recent ones
    for (int i = 1; i < chatHistory.Count - 5; i++)
    {
        if (chatHistory[i].Items?.Any(item => item is ImageContent) == true)
        {
            chatHistory.RemoveAt(i);
            i--; // Adjust index
        }
    }
}
```

#### 6. **Prompt Reinforcement**
For critical workflows, consider adding reminders in the conversation:
```csharp
// Every 5 interactions, add a subtle reminder
if (chatHistory.Count % 10 == 0)
{
    chatHistory.AddSystemMessage("Remember: Follow the tracking workflow strictly. Verify IDs visually before setting filters.");
}
```

## Testing Recommendations

### Test Scenario 1: Long Conversation
1. Have 15+ back-and-forth exchanges
2. Verify model still follows rules
3. Check response quality remains consistent

### Test Scenario 2: Image-Heavy Workflow
1. Request multiple image analyses
2. Monitor token usage
3. Verify no context overflow errors

### Test Scenario 3: Complex Multi-Step Tasks
1. Ask for tracking multiple objects
2. Verify each step is executed correctly
3. Check model doesn't skip workflow steps

## Model-Specific Notes

### Llama 3.2
- Context window: ~8K tokens typically
- Good with structured prompts
- Benefits from explicit step-by-step instructions
- Temperature 0.3-0.4 works well

### MiniCPM-V 8B
- Vision model with smaller context window
- Images consume ~500-1000 tokens each
- More sensitive to context overflow
- Consider keeping max 3-4 images in history
- Temperature 0.4-0.5 for better vision reasoning

## Monitoring Commands

Add these debug outputs to track health:

```csharp
Console.WriteLine($"📊 Context: {chatHistory.Count} messages");
Console.WriteLine($"📊 Estimated tokens: ~{EstimateTokenCount(chatHistory)}");
```

## Summary

The main issue was **aggressive context trimming** combined with **restrictive temperature settings**. Local models like Llama and MiniCPM need:
1. Sufficient conversation history (8-10 messages minimum)
2. Reasonable temperature (0.3-0.5) for flexibility
3. Strong, emphatic system prompts
4. Smart context management for vision workloads

The fixes applied should significantly improve consistency and rule-following behavior.
