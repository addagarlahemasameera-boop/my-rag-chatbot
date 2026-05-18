import boto3
import json

agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-west-2')

AGENT_ID = 'your_id'
AGENT_ALIAS_ID = 'your_id'

def lambda_handler(event, context):
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers(),
            'body': json.dumps('Preflight OK')
        }

    try:
        body = json.loads(event['body'])
        user_message = body.get('message', '')
        session_id = body.get('session_id', 'default-session')

        # Invoke Bedrock Agent
        response_stream = agent_runtime.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=user_message
        )['completion']

        final_response = ""
        citations = []

        for event in response_stream:
            if 'chunk' in event:
                chunk_bytes = event['chunk']['bytes']
                chunk_text = chunk_bytes.decode('utf-8')
                final_response += chunk_text

            elif 'trace' in event:
                trace_data = json.loads(event['trace']['output'].decode('utf-8'))
                if 'citations' in trace_data:
                    citations.extend(trace_data['citations'])

        return {
            'statusCode': 200,
            'headers': cors_headers(),
            'body': json.dumps({
                'response': {
                    'completion': {
                        'content': final_response,
                        'citations': citations
                    }
                }
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': cors_headers(),
            'body': json.dumps({ "error": str(e) })
        }

def cors_headers():
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }
