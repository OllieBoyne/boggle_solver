function myFunction(p1, p2) {
  return p1 * p2;   // The function returns the product of p1 and p2
}

// Load board from HTML
function getBoard(){
    var data = [];
    for (i=1;i<=5;i++){
        rowText = document.getElementById(i.toString()).value;
        if (rowText.length !== 5){
            console.error("Not all rows have length 5");
        }
        data.push(rowText) // toString(i)
    }
//
    return data
}

// Output value of board to check it is correct
function viewBoard(){
    loadTree();
    document.getElementById("demo").innerHTML = getBoard();
}

// Load tree from .json
function loadTree(){

    var loader = jQuery.getJSON();
    var tree = loader.load("tree.json");
    console.log(tree)
}