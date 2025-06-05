from main.lib.generic_api import GenericView
from ai.lib.retriever import HybridRetriever
from ai.models.document import DocumentChunk
from ai.models.conversation import Message, Conversation
from ai.serializers.conversation import MessageSerializer
from rest_framework.response import Response
from django.db import transaction
from rest_framework import status
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
from pydantic import ValidationError
import re
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

class RAGResponse(BaseModel):
    answer: str
    reason: str

class HybridRetrievalView(GenericView):
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    allowed_methods = ['create']
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def _retrieve(self, query):
        retriever = HybridRetriever()
        result = retriever.retrieve(query)
        return result

    @transaction.atomic
    def create(self, request):
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
        
        conversation = Conversation.objects.get(id=request.data.get("conversation_id"))
        
        message_history = conversation.messages.all().order_by('created_at')
        
        self.pre_create(request)
        
        print(f'RETRIEVING SIMILARITY FOR: {request.data.get("query")}')

        similar_text = self._retrieve(request.data.get("query"))
        
        document_chunks = DocumentChunk.objects.filter(id__in=[doc['id'] for doc in similar_text])
        
        context = "\n".join([f"Source: {doc['source']}\n\n{doc['text']}\n\n\n" for doc in similar_text])
        
        # Create system message with context
        system_prompt = f"""You are an expert in all information related to UP Cebu, which is a university in the Philippines. Based on the following documents from the university, answer the user's question.

Documents:
{context}

CRITICAL INSTRUCTIONS FOR RESPONSE FORMAT:
- You MUST respond ONLY with valid JSON
- Do NOT include any text before or after the JSON
- Do NOT include markdown code blocks or any formatting
- Your entire response must be parseable as JSON

CRITICAL INSTRUCTIONS FOR ANSWER FORMATTING:
- Format your answer in well-structured paragraphs with proper spacing
- Use bullet points (•) when listing unordered items
- Use numbers (1., 2., 3., etc.) when listing ordered items
- Use line breaks (\n) to separate paragraphs and sections
- Make the text easy to read and professional
- Use natural flowing text with proper transitions between ideas and paragraphs

Respond with this EXACT JSON structure:
{{
    "answer": "your detailed answer here with proper formatting using bullet points and paragraphs",
    "reason": "explain why you answered with this response and what context you used"
}}

Example response:
{{
    "answer": "To enroll at UP Cebu, you need to meet several requirements.\n\n• Pass the UPCAT (University of the Philippines College Admission Test)\n• Submit all required documents including transcripts and recommendation letters\n• Meet the minimum grade requirements for your chosen program\n\nThe admission process is competitive, so it's important to prepare thoroughly for the entrance examination.",
    "reason": "This information comes from the UP Cebu admissions document which outlines the entrance examination and requirements."
}}

REMEMBER: Your response must be valid JSON that can be parsed directly. Do not include any other text."""



        # Start with system message
        messages = [SystemMessage(content=system_prompt)]
        
        # Add message history
        for msg in message_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            # Skip system messages from history as we already have our context-aware system message
        
        # Add current user message
        messages.append(HumanMessage(content=request.data.get("query")))
        

        # Get response from LLM
        result = llm.invoke(messages)
        
        try:
            # First try to parse the response directly as JSON
            response_data = json.loads(result.content)
            structured = RAGResponse.model_validate(response_data)
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"\nDirect JSON parsing failed: {e}")
            print(f"Raw response: {result.content}")
            
            # Fallback: Try to extract JSON from the response
            try:
                # Look for JSON-like content between { and }
                json_match = re.search(r'\{.*?\}', result.content, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    print(f"Extracted JSON: {json_str}")
                    response_data = json.loads(json_str)
                    structured = RAGResponse.model_validate(response_data)
                else:
                    # Final fallback: create structured response from raw text
                    structured = RAGResponse(
                        answer=result.content,
                        reason="Response was not in proper JSON format, using raw LLM output"
                    )
                    
            except (json.JSONDecodeError, ValidationError) as fallback_error:
                print(f"Fallback parsing also failed: {fallback_error}")
                # Ultimate fallback
                structured = RAGResponse(
                    answer=result.content,
                    reason="Response could not be parsed as JSON, using raw LLM output"
                )
        
        # Create user message
        user_message = Message.objects.create(
            conversation=conversation,
            role="user",
            content=request.data.get("query")
        )
        user_message.context.set(document_chunks)
        
        # Create a new message for the assistant
        assistant_message = Message.objects.create(
            conversation=conversation,
            role="assistant",
            content=structured.answer
        )
        assistant_message.context.set(document_chunks)
        
        response = structured.model_dump()
        response["context"] = similar_text

        return Response(response, status=status.HTTP_200_OK)
