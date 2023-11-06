#------- run in Bash
# export FLASK_APP=hello         # To run the application, use the flask command or python -m flask. Before you can do that you need to tell your
                                 # terminal the application to work with by exporting the FLASK_APP environment variable:
# export FLASK_ENV=development   # To enable all development features, set the FLASK_ENV environment variable to development before calling flask run.

# flask run

# # ---------------------
# from flask import Flask, request

# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"
# # ---------------------

# # ---------------------
# from flask import Flask, request

# app = Flask(__name__)

# @app.route('/hello')
# def hello():
#     return 'Hello, Flask'

# # @app.route("/hi/<val>")
# # def hello_world(val):
# #     return "<p>Hello, World! </p>"+ val

# # @app.route("/add/<x>/<y>")
# # def sum_num(x,y):
# #     sum = int(x) + int(y)
# #     return str(sum)
# # ---------------------

# # ---------------------
# from flask import Flask, request

# app = Flask(__name__)

# @app.route("/model", methods=['POST'])
# def pred_model():
#     # Handle POST request with JSON data
#     js = request.get_json()
#     x = js['x']
#     y = js['y']
#     return str(x+y)
# # ---------------------

# # ---------------------
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route("/model", methods=['GET', 'POST'])
# def pred_model():
#     if request.method == 'GET':
#         # Handle GET request
#         x = int(request.args.get('x', 0))
#         y = int(request.args.get('y', 0))
#     elif request.method == 'POST':
#         # Handle POST request with JSON data
#         js = request.get_json()
#         x = js.get('x', 0)
#         y = js.get('y', 0)
#     else:
#         return jsonify({'error': 'Unsupported request method'}), 405

#     result = x + y
#     return jsonify({'result': result})

# if __name__ == "__main__":
#     app.run()

# # -----To make a GET request, open your web browser and navigate to 
# # http://127.0.0.1:5000/model?x=5&y=10

# # -----To make a POST request, you can use a tool like curl. For example
# # curl -X POST -H "Content-Type: application/json" -d '{"x": 5, "y": 10}' http://127.0.0.1:5000/model
# # ---------------------


from flask import Flask, request

app = Flask(__name__)

@app.route("/model", methods=['GET', 'POST'])
def pred_model():
    if request.method == 'GET':
        x = request.args.get('x', 0)
        y = request.args.get('y', 0)
    elif request.method == 'POST':
        js = request.get_json()
        # x = js['x']
        # y = js['y']
        x = js.get('x', 5)
        y = js.get('y', 6)
    else:
        return "Method not allowed", 405

    result = int(x) + int(y)
    return str(result)



# # ----------------not working-----
# import requests

# url = "http://127.0.0.1:5000/model"
# data = {"x": 5, "y": 10}

# response = requests.post(url, json=data)

# if response.status_code == 200:
#     result = response.json()
#     print("Response:", result)
# else:
#     print("Request failed with status code:", response.status_code)
# # ---------------------
