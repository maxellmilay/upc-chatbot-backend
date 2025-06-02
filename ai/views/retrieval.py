from main.lib.generic_api import GenericView
from ai.lib.retriever import HybridRetriever
from ai.models.document import DocumentChunk
from ai.models.conversation import Message, Conversation
from ai.serializers.conversation import MessageSerializer
from rest_framework.response import Response
from django.db import transaction
from rest_framework import status
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
from pydantic import ValidationError

class RAGResponse(BaseModel):
    answer: str
    reason: str

class HybridRetrievalView(GenericView):
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    allowed_methods = ['create']
    
    def _retrieve(self, query):
        retriever = HybridRetriever()
        result = retriever.retrieve(query)
        return result

    @transaction.atomic
    def create(self, request):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
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

You must only respond in JSON format with exactly this structure, and nothing else.:
{{
    "answer": "your detailed answer here",
    "reason": "Explain why you answered the question with the answer you did, and from what context did you use to answer your question."
}}"""



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
            # Parse the JSON response
            response_data = json.loads(result.content)
            structured = RAGResponse.model_validate(response_data)
            
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
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"\n❌ Failed to parse LLM response: {e}")
            print(f"Raw response: {result.content}")
            return Response(
                {"error": "Failed to parse response", "details": str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )
