import json
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

youtube_code = """<iframe width="560" height="315" src="https://www.youtube.com/embed/jDEYYZRWX_Q" title="YouTube 
video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; 
picture-in-picture; web-share" allowfullscreen></iframe>"""


@app.route("/")
def index():
    return render_template_string("<h1>Hello, Don't forget to check out my vlog channel<h1> <br>" + youtube_code)


@app.route("/check_api")
def data():
    query = request.args.get("query")
    if query:
        response = {"response": "Please Subscribe https://www.youtube.com/@adityamangal98"}
        json_data = json.dumps(response)
        return json_data
    else:
        response = {}
        json_data = json.dumps(response)
        return json_data


@app.route("/api/prompt_route")
def prompt_route():
    # user_prompt = request.form.get("user_prompt")
    user_prompt = request.args.get("user_prompt")
    if user_prompt:
        print(f'User Prompt: {user_prompt}')
        response = {"response": "Anand the best " + user_prompt}

        return jsonify(response), 200
    else:
        return "No user prompt received", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1111, debug=True)
