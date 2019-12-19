/*
    read iinput user image and show it.
*/
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imag_1')
                .attr('src', e.target.result)
                .width(150)
                .height(200);

        };

        reader.readAsDataURL(input.files[0]);
    }
}
/*
    show the scale check boxes.
*/
function scal_disply() {
    // Get the checkbox
    var checkBox = document.getElementById("SuperResolution");

    // If the checkbox is checked, display the scale check boxes
    if (checkBox.checked == true){
        $('.scale').each(function(){
            $(this).prop('disabled', false);
            })
    } else {
        $('.scale').each(function(){
            $(this).prop('disabled', true);
            })
    }
}


/*
    show the drop down if user want to apply denoising.
*/
function smoothingFactor_disaply() {
    // Get the checkbox
    var checkBox = document.getElementById("Denoising");

    // If the checkbox is checked, display the scale check boxes
    if (checkBox.checked == true){
        $('.smoothingFactor').prop('disabled', false);
            
    } else {
        $('.smoothingFactor').prop('disabled', true);
    }
}


/*
    check if the user choice an image or not.
*/
function validateForm(){
    flag = true;
    if(!document.getElementById('InputImage').value) {
       document.getElementById('msg_1').style.display = "block";
       flag =  false;
    }
    else {
        document.getElementById('msg_1').style.display = "none";
    }
    var checkBox1 = document.getElementById("Denoising");
    var checkBox2 = document.getElementById("SuperResolution");
    if ((checkBox1.checked == false) && (checkBox2.checked == false)) {
        document.getElementById('msg_2').style.display = "block";
        flag =  false;
    }
    else {
        document.getElementById('msg_2').style.display = "none";
    }
    return flag;
}



