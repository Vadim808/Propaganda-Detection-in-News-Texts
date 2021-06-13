console.log("content running");

let paragraphs = document.getElementsByTagName("p");

console.log("start");

var req = [], i = -1;
for (elt of paragraphs) {
	i += 1;
	if (i === 5) {
		break;
	}
	(function(i, elt){
		var text = elt.innerText;
		req[i] = new XMLHttpRequest();
		req[i].open('POST', "http://127.0.0.1:5000/", false);
	    req[i].setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
	    req[i].onreadystatechange = function() {
	    if (req[i].readyState == 4 && req[i].status === 200) {
	        	elt.innerHTML = req[i].responseText;
	    	}
		};
		req[i].send("name=" + text);
	})(i, elt);
}

console.log("end");