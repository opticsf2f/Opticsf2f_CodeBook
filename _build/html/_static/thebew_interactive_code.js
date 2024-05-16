function initInteractiveCode(){
    var allCode = document.getElementsByTagName("pre");
    for (var i=0; i<allCode.length; i+= 1){
        if ((window.getComputedStyle(allCode[i]).getPropertyValue("display")=="block" ||
            window.getComputedStyle(allCode[i]).getPropertyValue("display") == "flow-root")
            && allCode[i].parentElement.parentElement.parentElement.classList.contains("cell_input")){
            console.log(allCode[i]);
            console.log(allCode[i].parentElement.parentElement.parentElement.classList.toString());
            allCode[i].dataset.executable = "true";
            allCode[i].dataset.language = "python"
        }
    }

    thebelab.bootstrap(
        {
            requestKernel: true,
            mountActivateWidget: false,
            mountStatusWidget: true,
            useJupyterLite: true,
            useBinder: false,
            add_interrupt_button: false
          }
    )

}