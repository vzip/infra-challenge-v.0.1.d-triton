import argparse
import sys

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')

    FLAGS = parser.parse_args()
    try:
        triton_client = httpclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit(1)

    model_name = 'worker1'

    # There are seven models in the repository directory
    if len(triton_client.get_model_repository_index()) != 7:
        sys.exit(1)

    triton_client.load_model(model_name)
    if not triton_client.is_model_ready(model_name):
        sys.exit(1)

    # Request to load the model with override config in original name
    # Send the config with wrong format
    try:
        config = "\"parameters\": {\"config\": {{\"max_batch_size\": \"16\"}}}"
        triton_client.load_model(model_name, config=config)
    except InferenceServerException as e:
        if "failed to load" not in e.message():
            sys.exit(1)
    else:
        print("Expect error occurs for invald override config.")
        sys.exit(1)

    # Send the config with the correct format
    config = "{\"max_batch_size\":\"16\"}"
    triton_client.load_model(model_name, config=config)

    # Check that the model with original name is changed.
    # The value of max_batch_size should be changed from "8" to "16".
    updated_model_config = triton_client.get_model_config(model_name)
    if updated_model_config['max_batch_size'] != 16:
        print("Expect max_batch_size = 16, got: {}".format(
            updated_model_config['max_batch_size']))
        sys.exit(1)

    triton_client.unload_model(model_name)
    if triton_client.is_model_ready(model_name):
        sys.exit(1)

    # Trying to load wrong model name should emit exception
    try:
        triton_client.load_model("wrong_model_name")
    except InferenceServerException as e:
        if "failed to load" in e.message():
            print("PASS: model control")
    else:
        sys.exit(1)