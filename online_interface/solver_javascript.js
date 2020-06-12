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

// Load dict from .json
function loadDict(src){

    var client = new XMLHttpRequest();
    client.open("GET", src, async= false);
    client.onreadystatechange = function() {
    };
    client.send();
    return JSON.parse(client.responseText);

}

// Checks if {X, Y} coord is in Array of coords
function coordsInArray(X, Y, array){
    var entry;
    for(e=0; e<array.length;e++){
        entry = array[e];
        if((entry["X"] === X) && (entry["Y"] === Y)){
            return true
        }
    }
    return false
}

// Produces array of coordinates of possible next letters
// Does not allow for coordinates outside of the board, or repeated use of the same tile
function getNextMoves(x, y, width, used_tiles){
    var out = [];

    for (dx=-1; dx<2; dx++){
        for (dy=-1; dy<2; dy++) {
            var X = x + dx;
            var Y = y + dy;

            if ((0 <= X) && (X < width) && (0 <= Y) && (Y < width) && !(coordsInArray(X, Y, used_tiles))){
                out.push({X, Y});
            }
        }
    }
    return out
}

function solveBoard() {

    var tree = loadDict("tree.json");
    var dict = loadDict("CSW19.json")
    var board = getBoard();

    var width = board.length;

    // get all possible starting positions
    var x_starts = [];
    var y_starts = [];
    for (j=0;j<5;j++){
        for (i=0;i<5;i++) {
            x_starts.push(i);
            y_starts.push(j);
        }
    }

    var solutions = []; // List of all solutions
    var n_squares =  width ** 2;
    for (n=0; n<n_squares; n++){
        var x = x_starts[n];
        var y = y_starts[n];

        var c = board[y][x]; // Character
        if (c === "Q"){c = "QU"}
        else {}

        var prev_node = tree[c]; // Start moving through tree

        var temp_words = [[c, {"X":x, "Y":y}, [{"X":x, "Y":y}], prev_node ]]; // store temp words for comparing to
        var prev_words = []; // 'temp words' from previous iteration

        // vars changed each iter
        var next_moves = [];
        var word_data, next_node;
        var word_x, word_y;
        var word_string, used_tiles;
        var next_char, next_space;
        var new_string, new_path;
        var new_word_string, new_word;
        var temp_used_tiles; // used for each sub route

        // Search forwards, ending if no new 'temp words' found
        while (temp_words.length > 0) {
            prev_words = temp_words; // Duplicate array
            temp_words = []; // Clear temp words

            // For each previous word
            for (w=0;w<prev_words.length;w++){
                // Load word & metadata
                word_data = prev_words[w];
                word_string = word_data[0];
                word_x = word_data[1]["X"];
                word_y = word_data[1]["Y"];
                used_tiles = word_data[2];
                prev_node = word_data[3];

                next_moves = getNextMoves(word_x, word_y, width, used_tiles);

                // For each possible move, check validity
                for (n_move=0;n_move<next_moves.length;n_move++){
                    next_space = next_moves[n_move];
                    next_char = board[next_space["Y"]][next_space["X"]];
                    if (next_char==="Q"){next_char = "QU"}

                    next_node = prev_node[next_char]; // move through tree

                    // if valid character (in tree), and non-empty next node
                    //(next_char in prev_node) || !(isEmptyDict(next_node))
                    if (next_node !== undefined) {
                        new_word_string = word_string + next_char;
                        temp_used_tiles = used_tiles.slice();
                        temp_used_tiles.push(next_space);
                        new_word = [new_word_string, next_space, temp_used_tiles, next_node ];

                        temp_words.push(new_word);

                        // Check if this is a real word
                        // If either has no children, or is in dictionary
                        // for now, just has no children
                        if ((next_node.length === 0) || (new_word_string in dict)) {
                            if (!solutions.includes(new_word_string)){
                                solutions.push(new_word_string)
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort solutions according to length
    solutions = solutions.sort(function(a, b){
            return b.length - a.length;
        });

    return solutions
}

// Solve and add to GUI
function fullSolve(){
    var solution = solveBoard();
    var score = solution.reduce((a, b) => a + (b.length-3), 0); // Calculate score

    var dict_by_lengths = {};
    var length;
    // sort solutions by length
    for (w=0;w<solution.length;w++){
        word = solution[w];
        length = word.length;
        if (length in dict_by_lengths){
            dict_by_lengths[length].push(word)
        }
        else{
            dict_by_lengths[length] = [word]
        }
    }

    score_text = "Score: " + score.toString();

    document.getElementById("score").innerHTML = score_text;

    var breakdown = document.getElementById("breakdown");

    // clear any existing collapsibles
    breakdown.innerHTML = "";

    // Generate collapsibles
    for (const [ length, words] of Object.entries(dict_by_lengths).reverse()){
        // Make collapsible HTML element
        var collapsible = document.createElement("button");
        collapsible.innerHTML = length.toString() + " letter words";
        collapsible.setAttribute("class", "collapsible");
        breakdown.appendChild(collapsible);

        // Generate list of words
        var content = document.createElement("p");
        var text = document.createTextNode(words.join(", "));
        content.appendChild(text);
        breakdown.appendChild(content);
        content.style.display = "none"; // Initially, content is blocked

        // set collapsible to open/close content
        collapsible.addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                  content.style.display = "none";
                } else {
                  content.style.display = "block";
                }
              });


    }
}