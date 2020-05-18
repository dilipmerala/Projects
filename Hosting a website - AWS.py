#!/usr/bin/env python
# coding: utf-8

# # Hosting a static website

# In[ ]:


{
   "Version":"2012-10-17",
   "Statement":[
      {
         "Sid":"PublicReadGetObject",
         "Effect":"Allow",
         "Principal":"*",
         "Action":[
            "s3:GetObject"
         ],
         "Resource":[
            "arn:aws:s3:::example.com/*"
         ]
      }
   ]
}


# # EC2

# In[ ]:


chmod 400 NameOfFile.pem
sudo yum install -y httpd
start the web server. Run following command
sudo systemctl start httpd
#OPTIONAL
#Use the systemctl command to configure the Apache web server to start at each system boot.
sudo systemctl enable httpd
wget https://aws-training-utd-01.s3.amazonaws.com/wemeet.zip
unzip wemeet.zip


# # SNS Phone message

# In[ ]:


import json
import boto3
#print('Loading function')
region = 'us-east-1'
phoneNumber = '+12223334444'
msg = 'TestMessage'
snsClient = boto3.client('sns')
def lambda_handler(event, context):
    # TODO implement
    snsResponse = sendText(phoneNumber, msg)
    print(json.dumps(snsResponse))
    return snsResponse
# Funtion to send a text message to the customer. Number retrieved from DynamoDB table  
def sendText(phoneNumber, msg):
    #print("insideSendText")
    snsClient = boto3.client("sns", region)
    snsResponse = snsClient.publish(PhoneNumber = phoneNumber, Message = msg)
    #print(json.dumps(snsResponse))
    return snsResponse


# # SQS Phone message

# In[ ]:


import json
import boto3
queueName = 'testQueue-1'
region = 'us-east-1'
def lambda_handler(event, context):
    print("InputEvent:" + json.dumps(event))
    print("SQS Body:" + event['Records'][0]['body'])
    eventBody = json.loads(event['Records'][0]['body'])
    phone = eventBody['phone']
    text = eventBody['message']
    sendText(phone, text)
    return {
        'statusCode': 200,
        'body': json.dumps("Test Messgae")
    }
# Funtion to send a text message to the customer. Number retrieved from DynamoDB table  
def sendText(phoneNumber, msg):
    #print("insideSendText")
    snsClient = boto3.client("sns", region)
    snsResponse = snsClient.publish(PhoneNumber = phoneNumber, Message = msg)
    #print(json.dumps(snsResponse))
    return snsResponse


# # JSON for alerts

# In[ ]:


{
  "phone": "+19998887777",
  "message": "This Came from SQS"
}

