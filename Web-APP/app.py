import os, time, sys
import torchvision.utils as utils
import time


from flask import Flask, request, redirect, url_for, render_template, send_file
from FFDNET_test.test_ffdnet_ipol import test_ffdnet, estimate_noise
from PCARN_test.result import compute_image


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)



# the system has GPU or not
cuda = False

9
@app.route('/project')
def index ():
    return render_template("project.html")

@app.route('/demo', methods=['GET'])
def demo ():
    return render_template("demo.html")
@app.route('/test')
def test():
    return render_template("untitled.html")

@app.route('/demo/result', methods=['POST'])
def result ():
    user_adr = str(request.remote_addr)  + "_"
    target_in = os.path.join(APP_ROOT, 'static/input')
    if not os.path.isdir(target_in):
        os.mkdir(target_in)


    target_out = os.path.join(APP_ROOT, 'static/output')
    if not os.path.isdir(target_out):
        os.mkdir(target_out)
    
    # save the input image
    current_time = str(time.time())
    file = request.files.getlist("inputImage")[0]
    filename = file.filename
    input_path = "/".join([target_in, "in_" + user_adr + filename])
    output_path = "/".join([target_out, "out_" + current_time + user_adr + filename])
    # save the input image
    file.save(input_path)
    
    # run model Super resolution.
    if (request.form.get('superResolution')):
        scale = int(request.form['scale'])
        compute_image(input_path, scale, output_path)

    # run models denoising if choice.
    if (request.form.get('denoising') and request.form.get('superResolution')):
        smoothing_factor = request.form['smoothingFactor']

        if (smoothing_factor == 'Auto'):
            smoothing_factor = estimate_noise(output_path)
        else:
            smoothing_factor = float(smoothing_factor)

        test_ffdnet (output_path, output_path, cuda, smoothing_factor)
        
    elif (request.form.get('denoising')):
        smoothing_factor = request.form['smoothingFactor']

        if (smoothing_factor == 'Auto'):
            smoothing_factor = estimate_noise(input_path)
        else:
            smoothing_factor = float(smoothing_factor)

        test_ffdnet (input_path, output_path, cuda, smoothing_factor)
            

    return render_template("result.html", input = "in_" + user_adr + filename, output = "out_"+ current_time + user_adr + filename)




if __name__ == "__main__":
    app.run(debug=True)
