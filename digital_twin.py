#!/usr/bin/env python3
import pika
import json

# Create connection to rabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Create exchange and queue
channel.exchange_declare(exchange='fmi_digital_twin', exchange_type='direct')
result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue
channel.queue_bind(exchange='fmi_digital_twin', queue=queue_name,
                   routing_key='sim1.data.from_physical_twin')

# Accumulated received data
received_data = []

# Callback per ricevere e salvare dati
def callback(ch, method, properties, body):
    try:
        if "waiting for input data for simulation" in str(body):
            return
        message = json.loads(body)

        lead_accel = message.get('lead_accel')
        timestamp = message.get('time')
        if lead_accel is not None and timestamp is not None:
            new_msg = {
                'time': timestamp,
                'lead_acceleration': lead_accel
            }
            #print(" [x] Pubblico:", new_msg)
            channel.basic_publish(
                exchange='fmi_digital_twin',
                routing_key='sim2.data.to_DT',
                body=json.dumps(new_msg)
            )
        else:
            print(" [!] Messaggio ricevuto incompleto:", message)
    except Exception as e:
        print(" [!] Errore nel parsing del messaggio:", e)
        #print(" [x] Ricevuto:", message)

# Setup consume
channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

# Start consuming
print(" [*] In ascolto... premi CTRL+C per terminare.")
try:
    channel.start_consuming()
except KeyboardInterrupt:
    print("\n [*] Interruzione da tastiera.")
finally:
    connection.close()
