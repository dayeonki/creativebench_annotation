import base64, requests, configparser, os, argparse, json, random
from openai import AzureOpenAI

current_path = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(current_path, '../config/config.ini') # see '../../README.md' for instructions on creating the configuration file
if not os.path.exists(CONFIG_FILE):
    CONFIG_FILE = os.path.expanduser('~/azure_openai/config.ini')
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Error: Configuration file not found: {CONFIG_FILE}")

# DEFAULT_CONFIG_LABEL = ['BB-GPT4v'] # add your configuration label here, e.g. ['default', 'gpt4-vision-preview']
DEFAULT_CONFIG_LABEL = ['BB-o1'] # add your configuration label here, e.g. ['default', 'gpt4-vision-preview']
# DEFAULT_CONFIG_LABEL = ['BB-o3-mini'] # add your configuration label here, e.g. ['default', 'gpt4-vision-preview']
# DEFAULT_CONFIG_LABEL = ['epic-cgi-o1'] # add your configuration label here, e.g. ['default', 'gpt4-vision-preview']
# DEFAULT_CONFIG_LABEL = ['BB-GPT4o'] # add your configuration label here, e.g. ['default', 'gpt4-vision-preview']

DEFAULT_SYS_PROMPT = "You are trained to interpret images about people and make responsible assumptions about them."
MAX_IMAGE_COUNT = 2


class AoaiGptInterface:
    """
    Interface to call AOAI GPT.
    """
    def __init__(self, CONFIG_LABEL = DEFAULT_CONFIG_LABEL):
        self.config_labels = CONFIG_LABEL
        self.config = configparser.ConfigParser()
        self.config.read(CONFIG_FILE)

    def select_config(self):
        """
        Selects a configuration from the config file.
        """
        try:
            chosen_label = random.choice(self.config_labels)
            self.api_base = self.config.get(chosen_label, 'api_base')
            self.api_key = self.config.get(chosen_label, 'api_key')
            self.api_version = self.config.get(chosen_label, 'api_version')
            self.deployment_name = self.config.get(chosen_label, 'engine')
        except Exception as e:
            raise Exception(f"Error reading configuration file: {e}")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}"
        )
        # print(f"Base URL: {self.api_base}")

    def encode_image(self, image_path):
        """
        Encodes an image to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('ascii')

    def call_vision_api(self, image_path, prompt, sys_prompt=DEFAULT_SYS_PROMPT):
        """
        Calls AOAI GPT using the OpenAI Python API.
        """
        self.select_config()
        encoded_image_list = []
        if isinstance(image_path, list):
            for img in image_path:
                if len(encoded_image_list) >= MAX_IMAGE_COUNT:
                    print(f"Warning: Only the first {MAX_IMAGE_COUNT} images will be processed.")
                    break
                encoded_image_list.append(self.encode_image(img))
        elif isinstance(image_path, str):
            if os.path.isdir(image_path):
                for img in os.listdir(image_path):
                    if img.endswith(('.png', '.jpg', '.jpeg')):
                        if len(encoded_image_list) >= MAX_IMAGE_COUNT:
                            print(f"Warning: Only the first {MAX_IMAGE_COUNT} images will be processed.")
                            break
                        encoded_image_list.append(self.encode_image(os.path.join(image_path, img)))
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Error: Image file not found: {image_path}")
                encoded_image_list.append(self.encode_image(image_path))
        else:
            raise ValueError(f"Error: Invalid image path: {image_path}")
        
        user_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        for encoded_image in encoded_image_list:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            )

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                { "role": "system", "content": sys_prompt},
                { "role": "user", "content": user_content} 
            ],
            # temperature=0,
            seed=0,
            # max_tokens=4096 
        )
        return response.model_dump()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calling GPT Vision API.")
    parser.add_argument("-i", "--image", type=str, help="Path to the image file/dir/list of image paths.", default="tests/winter.png", nargs='?')
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to describe the task for the GPT-4 Vision API.", default="Describe the details in the image, and if multiple images provided, describe each of them.", nargs='?')
    args = parser.parse_args()
    gpt_interface = AoaiGptInterface()

    try:
        print(f"Calling AOAI GPT with config label: {gpt_interface.config_labels}")
        response_dict = gpt_interface.call_vision_api(args.image, args.prompt)
        # print(json.dumps(response, indent=4))
        print(json.dumps(response_dict, indent=4))
    except Exception as e:
        print(f"Error: {e}")

