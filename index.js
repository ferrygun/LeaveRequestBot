const fs = require('fs');

const {
    App
} = require("@slack/bolt");
const axios = require('axios');

const app = new App({
    token: 'xoxb-',
    signingSecret: '',
    socketMode: true, // enable to use socket mode
    appToken: 'xapp-'
});

function escapeNewlines(inputString) {
  return inputString.replace(/\n/g, "");
}


let URL = 'http://<>:8000';

let existing_leavesArr = [];

app.message(async ({
    message,
    say
}) => {
    try {

        // Define the payload to send in the POST request
        let user_id = 'ferry.djaja';

        const payload = {
            userMessage: {
                user_id: user_id, //message.user
                question: message.text
            },
        };

        // Make an HTTP POST request to http://localhost:8000/ask
        console.log('post request: ' + payload.userMessage.question)
        const response = await axios.post(URL + '/ask', payload);

        // Check if the request was successful and respond accordingly
        if (response.status === 200) {
            let responseBody = escapeNewlines(response.data); // Get the response body
            responseBody = JSON.parse(responseBody);

			console.log("responseBody: ")
            console.log(responseBody)


            if (responseBody.hasOwnProperty('existing_leaves') ) {
            	
            }

            if(responseBody.complete === true) {

            	// Convert the conversationRecord data to JSON format
				const jsonData = JSON.stringify(responseBody.existing_leaves, null, 2);

				// Specify the file path where you want to save the data
				const filePath = "/GPT/leave_" + user_id + ".txt"

				// Write the JSON data to the file
				fs.writeFile(filePath, jsonData, (err) => {
					if (err) {
						console.error('Error writing to file:', err);
					} else {
					    console.log('Data written to file successfully.');
					}
				});

				
				console.log('--Reset Session--')
				const payload1 = {
				    "user_id": user_id
				}

            	//reset the session:
            	const response1 = await axios.post(URL + '/reset', payload1);
            	console.log(response1.status);

            }

            //response
            say(responseBody.answer);
        } else {
            say('Hello Human! HTTP POST request failed.');
        }
    } catch (error) {
        console.error(error);
        say('Hello Human! An error occurred while sending the HTTP POST request.');
    }
});

(async () => {
    const port = 3000
    await app.start(process.env.PORT || port);
    console.log('Bolt app started!!');
})();
