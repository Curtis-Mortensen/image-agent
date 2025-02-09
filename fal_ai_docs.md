The client API handles the API submit protocol. It will handle the request status updates and return the result when the request is completed.

import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

result = fal_client.subscribe(
    "fal-ai/fast-lightning-sdxl",
    arguments={
        "prompt": "photo of a girl smiling during a sunset, with lightnings in the background"
    },
    with_logs=True,
    on_queue_update=on_queue_update,
)
print(result)   