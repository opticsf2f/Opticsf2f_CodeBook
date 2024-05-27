function initInteractiveCode(){
    var allCode = document.getElementsByTagName("pre");
    for (var i=0; i<allCode.length; i+= 1){
        if ((window.getComputedStyle(allCode[i]).getPropertyValue("display")=="block" ||
            window.getComputedStyle(allCode[i]).getPropertyValue("display") == "flow-root")
            ){
            if (allCode[i].parentElement.parentElement.parentElement.classList.contains("cell_input")){
                allCode[i].dataset.executable = "true";
                allCode[i].dataset.language = "python";
                var parent = allCode[i].parentElement.parentElement.parentElement;
                parent.style.borderWidth="0px 0px 0px 3px";
                parent.style.borderColor="#fe7b3a";
                parent.style.borderRadius="0px";
            }
            else if (allCode[i].parentElement.parentElement.parentElement.classList.contains("cell_output")){
                allCode[i].style.display="None"; 
            }
        }  

    }

    allCode = document.getElementsByClassName("cell_output");
    for (var i=0; i<allCode.length; i+= 1){
        allCode[i].style.display="None"; 
    }



    thebelab.bootstrap(
        {
            requestKernel: true,
            mountActivateWidget: false,
            mountStatusWidget: true,
            useJupyterLite: true,
            useBinder: false,
            add_interrupt_button: true,
            codeMirrorConfig: {
                theme: "one-dark",
                viewportMargin: 150,
                lineWrapping: true
            },
          }
    )
    adjustCSSRules(".CodeMirror", "height: auto");
    adjustCSSRules(".CodeMirror", "background: #282c34");
    adjustCSSRules(".CodeMirror", "font-family: 'Jet Brains Mono Nerd', monospace;");
    adjustCSSRules(".CodeMirror-wrap", "border-radius: '5px'");
}

function adjustCSSRules(selector, props, sheets){

    // get stylesheet(s)
    if (!sheets) sheets = [...document.styleSheets];
    else if (sheets.sup){    // sheets is a string
        let absoluteURL = new URL(sheets, document.baseURI).href;
        sheets = [...document.styleSheets].filter(i => i.href == absoluteURL);
        }
    else sheets = [sheets];  // sheets is a stylesheet

    // CSS (& HTML) reduce spaces in selector to one.
    selector = selector.replace(/\s+/g, ' ');
    const findRule = s => [...s.cssRules].reverse().find(i => i.selectorText == selector)
    let rule = sheets.map(findRule).filter(i=>i).pop()

    const propsArr = props.sup
        ? props.split(/\s*;\s*/).map(i => i.split(/\s*:\s*/)) // from string
        : Object.entries(props);                              // from Object

    if (rule) for (let [prop, val] of propsArr){
        // rule.style[prop] = val; is against the spec, and does not support !important.
        rule.style.setProperty(prop, ...val.split(/ *!(?=important)/));
        }
    else {
        sheet = sheets.pop();
        if (!props.sup) props = propsArr.reduce((str, [k, v]) => `${str}; ${k}: ${v}`, '');
        sheet.insertRule(`${selector} { ${props} }`, sheet.cssRules.length);
        }
    }