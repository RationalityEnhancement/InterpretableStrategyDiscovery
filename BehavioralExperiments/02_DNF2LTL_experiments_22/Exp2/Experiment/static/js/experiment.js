var CONDITION, DEBUG, JUMP_TO_BLOCK, PARAMS, BONUS, SCORE, STRUCTURE_TRAINING, TRIALS_INNER_REVEALED, TRIALS_TRAINING,
    TRIALS_ACTION, NUM_TRAINING_TRIALS, NUM_TEST_TRIALS_BEFORE_ATTENTION, NUM_TEST_TRIALS_AFTER_ATTENTION, calculateBonus,
    createStartButton, delay, getActionTrials, getTrainingTrials,
    initializeExperiment, loadTimeout, psiturk, saveData, slowLoad, getIncreasingSmall, getIncreasingLarge, getRoadtrip,
    TRIALS_INCREASING_SMALL, TRIALS_ROADTRIP, STRUCTURE_ROADTRIP, TRIALS_INCREASING_LARGE, TRIALS_MORTGAGE, getMortgage, toggleQuizInfo, with_instructions, order, SECRET_COMPLETION_CODE;


CONDITION =  parseInt(condition);
with_instructions = [2,3].includes(CONDITION);
with_roadtrip_first = [1,3].includes(CONDITION);
DEBUG =  false;

//DEBUG =  parseInt(window.prompt('debug 0-1', 1)) == 1;
//with_instructions = parseInt(window.prompt('with instructions?  0-1', 1));
//with_roadtrip_first = parseInt(window.prompt('roadtrip first? 0-1', 1));

//with_instructions = true;
//with_roadtrip_first = true;

PAY_BASE = '£1.50';
PAY_MEAN = '£2.25';
SECRET_COMPLETION_CODE = '23EB2255'

JUMP_TO_BLOCK = 0;

TRIALS_TRAINING = void 0;

TRIALS_INNER_REVEALED = void 0;

STRUCTURE_TRAINING = void 0;

SCORE = 0; // mortgage
BONUS = 0; // roadtrip

MAX_AMOUNT = 10;

REPETITIONS = 0; //tracks trials in instructions quiz

calculateBonus = void 0;

getTrainingTrials = void 0;

getActionTrials = void 0;

trialCount = 0;

psiturk = new PsiTurk(uniqueId, adServerLoc, mode);
psiturk.recordUnstructuredData('condition', CONDITION);

PARAMS = {
    inspectCost: 1,
    bonusRate: 0.005, // 5 cent for each correct mortgage
    CODE: ['SALAMANDERV2'],
    startTime: Date(Date.now())
};

saveData = function () {
    return new Promise(function (resolve, reject) {
        var timeout;
        timeout = delay(10000, function () {
            return reject('timeout');
        });
        return psiturk.saveData({
            error: function () {
                clearTimeout(timeout);
                console.log('Error saving data!');
                return reject('error');
            },
            success: function () {
                clearTimeout(timeout);
                console.log('Data saved to psiturk server.');
                return resolve();
            }
        });
    });
};

createStartButton = function () {
    initializeExperiment();
};

$(window).on('beforeunload', function () {
    return 'Are you sure you want to leave?';
});

$(window).resize(function () {
    return checkWindowSize(800, 600, $('#jspsych-target'));
});

$(window).resize();

$(window).on('load', function () {
    // Load data and test connection to server.
    slowLoad = function () {
        var ref;
        return (ref = $('slow-load')) != null ? ref.show() : void 0;
    };
    loadTimeout = delay(12000, slowLoad);
    psiturk.preloadImages(['static/images/spider.png']);
    return delay(300, function () {
        console.log("SELECTED CONDITION", CONDITION);
        TRIALS_ROADTRIP = loadJson('static/json/roadtrip_trials_singleLow.json');
        STRUCTURE_ROADTRIP = loadJson('static/json/structure/31123.json');
        TRIALS_MORTGAGE = loadJson('static/json/mortgage.json');

        getRoadtrip = (function () {
            var idx, t;
            t = _.shuffle(TRIALS_ROADTRIP);
            idx = 0;
            return function (n) {
                idx += n;
                return t.slice(idx - n, idx);
            };
        })();
        getMortgage = (function () {
            var idx, t;
            t = _.shuffle(TRIALS_MORTGAGE);
            idx = 0;
            return function (n) {
                idx += n;
                return t.slice(idx - n, idx);
            };
        })();

        return saveData().then(function () {
            clearTimeout(loadTimeout);
            return delay(500, createStartButton);
        }).catch(function () {
            clearTimeout(loadTimeout);
            return $('#data-error').show();
        });
    })
})

createStartButton = function () {
    initializeExperiment();
};

initializeExperiment = function () {
    var experiment_timeline
    $('#jspsych-target').html('');


    // ---------------------------------------------------------------------------------------------
    // ------------------------------------------- INTRO --------------------------------------------
    // ---------------------------------------------------------------------------------------------

    let welcome = {
        type: 'instructions',
        show_clickable_nav: true,
        pages: function(){
          return [
          `<h1>Structure of the experiment</h1>
          <ul style='text-align: left;'>
            <li>This experiment consists of two parts. In each part you will go through two phases:</li>
            <ul>
              <li><b>Instructions</b>: In this phase, you will be introduced to the experiment and given instructions about it.</li>
              <li style='margin-bottom:0.4em'><b>Playing a game</b>: In this phase, you will apply the knowledge acquired in the instructions phase in a special game.</li>
            </ul>
            <li style='margin-bottom:0.4em'>If you complete the experiment, you will receive a base pay of ` +PAY_BASE+ ` and a bonus which is dependent on your performance in the game. On average, a person who conscientiously follows the instructions receives ` + PAY_MEAN+ ` in total.</li>
            <li> Click <b>Next</b> to start the first part of the experiment. </li>
          </ul>

          `,

          `<h1> Game 1: `+ (with_roadtrip_first ? "Travel planner" : "Mortgage Choice") + `</h1>
          <div style="text-align: left">
          We will now introduce you to the first game.
          </div>`]
      }
    }

    let divider = {
      type: 'instructions',
      show_clickable_nav: true,
      pages: function () {
          return [
              `<h1> Game 2: `+ (with_roadtrip_first ? "Travel planner" : "Mortgage Choice") + ` </h1>
              <div style="text-align: left">
                  <li>We will now introduce you to the second game.</li>
              </div>`
            ]
      }
    }

    toggleQuizInfo = function(block) {

       if(block == 'roadtrip'){
         infos = ["You will play a game where you pretend to be a travel planner.",
         "The client must get from a start city to one of the cities with airports.",
         "At any time, you can select parts of the route by clicking on the lines connecting the cities.",
         "Each night they must stay in a hotel, which costs money.",
         "You want to find a cheap route for your client.",
         "You can look up the price of the cheapest hotel in a city by typing the city name in a text box and clicking Reveal",
         "Airport hotels start at <strong>$20</strong>.",
         "Revealing the price costs <strong>$10.</strong>",
         "You do not need to check the prices of every city on the route before submitting."
         ]

       } else if(block == 'mortgage'){
         infos = ["You have found your dream property and want to ask the bank for a loan.",
         "The bank presents you with <strong>three</strong> different mortgage plans.",
         "Each mortgage plan has <strong>three</strong> different interest rates.",
         "Unfortunately, the bank clerk forgot to tell you about the interest rates.",
         "You decide to call the bank to ask about the interest rates.",
         "However, the bank clerk only has time to tell you <strong>up to three</strong> interest rates.",
         "You can click up to <strong>3</strong> times after which you have to make a decision which mortgage plan to choose.",
         "You can select a mortgage plan at any time by clicking on the grey mortgage plan button (A, B or C).",
         ]
       }

       let div = document.getElementById("quizInfoDiv");
       let button = document.getElementById("quizInfoButton");

       if (div.innerHTML === "") {

         let content = '<ul>';
         for(q in infos){
           content = content + "<li>" + infos[q] + "</li>";
         }
         content = content + "</ul>"

         button.innerHTML = 'Hide instructions';
         div.innerHTML = content;
       } else {
         button.innerHTML = 'Show instructions'
         div.innerHTML = "";
       }
    }

    // ---------------------------------------------------------------------------------------------
    // ------------------------------------------- ROADTRIP --------------------------------------------
    // ---------------------------------------------------------------------------------------------

    let roadtrip_instructions = {
    type: 'instructions',
    show_clickable_nav: true,
        pages: function () {
        return [
            `<h1> Travel planner: Intro </h1>
                <div style="text-align: left; margin-bottom: -8em">
                    <li>In the Travel Planner game you pretend to be a travel planner.</li>
                    <li>You start by seeing a map like shown below.</li>
                    <li>Your client needs to travel from the city with the car (Ruby Ridge) to one of the cities with an airport.</li>
                    <li>Getting from city A to city B is only possible when there is an arrow from city A to city B.</li>
                    <br>
                    <img className='display' style="width:60%; height:40%"
                         src='static/images/roadtrip/instructions/Instructions-screenshot-with-car-crop.png'/>
                </div>
                `,
            `<h1> Travel planner: Instructions </h1>
                <div style="text-align: left">
                    <li>Your client can travel only one city per day. During the night he or she stays in a hotel, which costs money.</li>
                    <li>Your client wants a morning flight, so they must pay for a hotel in the airport city as well.</li>
                    <li>The price of the hotel varies between the different cities.</li>
                    <li>Airport hotels start at $20.</li>
                    <li>Your client is on a tight budget of $500 and wishes to take the cheapest route.</li>
                    <li>Your goal is to choose which cities to traverse so that the price of the trip was as cheap as possible.</li>
                    <br>
                    <img className='display' style="width:60%; height:auto;  margin-bottom: -8em"
                         src='static/images/roadtrip/instructions/Instructions-screenshot-with-all-prices-crop.png'/>
               </div>
            `,
            `<h1> Travel planner: Price Check </h1>
                <div style="text-align: left">
                    <li>You can look up the price of the cheapest hotel in a city by typing the city name in a text box and clicking Reveal.</li>
                    <li>The prices are negative to convey the cost you will incur by staying in the city.</li>
                    <li>When you look up a city, its price is revealed on the map.</li>
                    <li>Revealing the price costs <strong>$10.</strong></li>
                    <br>
                    <img className='display' style="width:60%; height:auto"
                         src='static/images/roadtrip/instructions/Instructions-screenshot-with-city-name-crop.png'/>
               </div>
            `,
            `<h1> Travel planner: Summary </h1>
                <div style="text-align: left">
                    <li>At any time, you can select parts of the client's route by clicking on the arrows.</li>
                    <li>If you change your mind, you can unselect arrows by clicking them again.</li>
                    <li>When you have finalized your route, click Submit. </li>
                    <li>You do not need to check the prices of every city on the route before submitting.</li>
                    <br>
                    <img className='display' style="width:60%; height:auto"
                         src='static/images/roadtrip/instructions/Instructions-screenshot-with-submit-crop.png'/>
               </div>

            `,
            `<h1> Quiz </h1>

                Before you can begin playing the game, you must pass the quiz to show that you understand the rules.
                If you get any of the questions incorrect, you will be brought back to the instructions to review and try the quiz again.`
        ];
        }
    };

    let roadtrip_procedural = {
      type: 'instructions',
      show_clickable_nav: true,
      pages: function(){
        return with_instructions ? [markdown(`<h1>Advice for scoring a high bonus</h1> To help you score higher in the roadtrip planner game, we will show you its near-optimal strategy. This strategy describes in what order to explore the hotel prices. Please take a moment to understand this advice and how you could apply it in the game: <br><p style='color: #c73022'> Look up the prices of the most distant hotels that you have not looked up yet.<br>Repeat this step until all the distant hotels' prices are looked up or you have encountered the lowest possible hotel price. </p>`)] : [markdown("<h1>Instructions</h1> Good luck in finding a cheap path!")]
      }
    }

    let roadtrip_quiz = {
        preamble: function () {
            return markdown(`<h1>Quiz</h1>Please answer the following questions about the game. Questions marked with (*) are compulsory. <br><br>
              <button id="quizInfoButton" class="btn btn-primary" onclick="toggleQuizInfo('roadtrip')">
                Show instructions</button> <div id="quizInfoDiv" style="width:800px;text-align:left;margin: 0 auto;"></div>`);
        },
        type: 'survey-multi-choice',
        questions: [
            {
                prompt: "What are you (the travel planner) looking for?",
                options: ['The shortest route for my client.', 'A cheap route for my client.', 'Any route.'],
                horizontal: false,
                required: true
            },
            {
                prompt: "What is the cheapest possible price for an airport hotel?",
                options: ['$20', '$150', '$200'],
                horizontal: false,
                required: true
            },
            {
                prompt: "How do you select and unselect a route?",
                options: ['By clicking on the cities.', 'By clicking on the connecting line between the cities.', 'By typing the route into the box.'],
                horizontal: false,
                required: true
            }
        ],
        data: {
            correct: {
                Q0: 'A cheap route for my client.',
                Q1: '$20',
                Q2: 'By clicking on the connecting line between the cities.'
            }
        }
    };

    let roadtrip_instruct_loop = {
        timeline: [roadtrip_instructions, roadtrip_quiz],
        loop_function: function (data) {
            var i, responses, corrects;
            responses = data.last(1).values()[0].response;
            corrects = data.last(1).values()[0].correct;

            for (i in responses) {
                if ((responses[i] != corrects[i]) && !DEBUG) {
                    REPETITIONS += 1;
                    if(REPETITIONS >= 3){
                      psiturk.recordUnstructuredData('exclusion', true);
                      psiturk.saveData();

                      if(with_roadtrip_first){
                        $('#jspsych-target').html(`<h1>Please return your submission</h1>\n<p>\nThank you for your interest in this study.
                          However, it is crucial that you understand these instructions to take part, and our data indicate that this is not the case.
                          I'm afraid you cannot proceed further in the experiment as a result.
                          Please return your submission on Prolific now by selecting the "Stop without completing" button, to avoid receiving a rejection.`);

                      } else {
                        $('#jspsych-target').html(`<h1>End of study</h1>\n<p>\nThank you for your interest in this study.
                          However, it is crucial that you understand these instructions to take part, and our data indicate that this is not the case.
                          I'm afraid you cannot proceed further in the experiment as a result and will not receive any bonus.<br><br>
                          <strong>To receive the base pay please enter the secret completion code on prolific:  ` + SECRET_COMPLETION_CODE + "</strong>");
                      }
                    }
                    else{
                      console.log('wrong way2');
                      alert(`You got at least one question wrong. We'll send you back to the instructions and then you can try again.`);
                      return true; // try again
                    }
                }
            }
            REPETITIONS = 0;
            //psiturk.recordUnstructuredData('exclusion', false);
            psiturk.finishInstructions();
            psiturk.saveData();
            return false;
        }
    };

    let roadtrip_intro = {
      type: 'instructions',
      show_clickable_nav: true,
      pages: function () {
        return [
            `<h1> Gameplay phase </h1>
            You will now play 8 rounds of the Travel planner game.
            Remember, the better you perform, the higher your bonus will be. Have Fun!
            </div>`
            ]
          }
      }

    let roadtrip_trials = {
      type: 'roadtrip',
      blockName: 'test',
      display_element: "jspsych-target",
      timeline: DEBUG ? getRoadtrip(2) : getRoadtrip(8),
      show_instructions: with_instructions,
      instructions_html: `<div id="instructions" style="text-align: center; font-weight: 700; position: absolute; top: 15px; z-index: 5; width:90%; left: 7vw">Advice for scoring a high bonus: <p style="text-align: center; font-weight: 700; color: #c73022"> Look up the prices of the most distant hotels that you have not looked up yet.<br>Repeat this step until all the distant hotels' prices are looked up or you have encountered the lowest possible hotel price.</p></div>`
    }

    let roadtrip_testblock = {
      timeline: with_instructions ?  [roadtrip_procedural, roadtrip_intro, roadtrip_trials] : [roadtrip_intro, roadtrip_trials]
    }

    // ---------------------------------------------------------------------------------------------
    // ------------------------------------------- MORTGAGE --------------------------------------------
    // ---------------------------------------------------------------------------------------------

    let mortgage_instructions = {
        type: 'instructions',
        show_clickable_nav: true,
        pages: function () {
            return [
                `<h1> Mortgage game: Intro</h1>
                <div class='instructions'>
                    <li>In the Mortgage game you have found your dream property and want to ask the bank for a loan.</li>
                    <li>The bank presents you with <strong>three</strong> different mortgage plans.</li>
                    <li>Each mortgage plan has <strong>three</strong> different interest rates:
                      <ul style='padding-left: 80px; list-style-type: "-- ";'>
                      <li> one for the 1st year (2022) </li>
                      <li> one for the 2nd until 5th year (2023 – 2027) </li>
                      <li> one for the 6th until 30th year (2028 – 2052)</li>
                      </ul>
                    </li>
                </div>

                `,
                `<h1> Mortgage game: Interest rates</h1>
                <div class='instructions'>
                    <li>Unfortunately, the bank clerk forgot to tell you about the interest rates.</li>
                    <li>In the example below, you can see three plans (Mortgage plan A, Mortgage plan B, Mortgage plan C) but their corresponding interest rates are hidden underneath the blue fields.</li>
                    <img class='center-img' src='static/images/mortgage/mortgage-example.png'/>
                </div>

                `,
                `<h1> Mortgage game: Rules </h1>
                <div class='instructions'>
                    <li>You decide to call the bank to ask about the interest rates.</li>
                    <li>However, the bank clerk only has time to tell you <strong>up to three</strong> interest rates.</li>
                    <li>Each time a bank clerk tells you about the interest rate corresponds to one click.</li>
                    <li>That means you can only click <strong>up to 3</strong> times.</li>
               </div>`,

                `<h1> Mortgage game: Example </h1>
                <div class='instructions'>
                    <li>Below you will see an example with one field revealed.</li>
                    <li> In the example, the interest rate from 2023 to 2027 for mortgage plan  B was revealed. </li>
                    <li> In the example, you would have to pay 1.61% interest rate in each of the 4 years when you select mortgage plan B.
                    <img class='center-img' src='static/images/mortgage/mortgage-example-revealed.png'/>
               </div>`,

                `<h1> Mortgage game: Scoring </h1>
                <div class='instructions'>
                    <li>You can click up to <strong>3</strong> times after which you have to make a decision which mortgage plan to choose.</li>
                    <li>You can select a mortgage plan at any time by clicking on the grey mortgage plan button (A, B or C).</li>
               </div>`
            ];
        }
    };

    let mortgage_quiz = {
      preamble: function () {
          return markdown(`<h1>Quiz</h1>Please answer the following questions about the game. Questions marked with (*) are compulsory. <br><br>
            <button id="quizInfoButton" class="btn btn-primary" onclick="toggleQuizInfo('mortgage')">
              Show instructions</button> <div id="quizInfoDiv" style="width:800px;text-align:left;margin: 0 auto;"></div>`);
      },
        type: 'survey-multi-choice',
        questions: [
            {
                prompt: "When choosing a mortgage plan, you want a plan with...",
                options: ['a high interest rate.', 'a low interest rate.'],
                horizontal: false,
                required: true
            },
            {
                prompt: "How many different interest rates does each mortgage plan has?",
                options: ['1', '2', '3'],
                horizontal: false,
                required: true
            },
            {
                prompt: "How often can you click at most to find out about the three different interest rates of the three different plans?",
                options: ['1 time',
                    '2 times',
                    '3 times',
                    '4 times'],
                horizontal: false,
                required: true
            },
            {
                prompt: "What is the interest rate for the last 25 years of the three mortgage plans?",
                options: ['1%',
                    '1.5%',
                    'Unknown but can be found out by clicking the corresponding field.'],
                horizontal: false,
                required: true
            },
        ],
        data: {
            correct: {
                Q0: 'a low interest rate.',
                Q1: '3',
                Q2: '3 times',
                Q3: 'Unknown but can be found out by clicking the corresponding field.'
            }
        }
    };

    let mortgage_instruct_loop = {
        timeline: [mortgage_instructions, mortgage_quiz],
        loop_function: function (data) {
            var i, responses, corrects;
            responses = data.last(1).values()[0].response;
            corrects = data.last(1).values()[0].correct;

            for (i in responses) {
                if ((responses[i] != corrects[i]) && !DEBUG) {
                    REPETITIONS += 1;
                    if(REPETITIONS >= 3){
                      psiturk.recordUnstructuredData('exclusion', true);
                      psiturk.saveData();

                      if(!with_roadtrip_first){
                        $('#jspsych-target').html(`<h1>Please return your submission</h1>\n<p>\nThank you for your interest in this study.
                          However, it is crucial that you understand these instructions to take part, and our data indicate that this is not the case.
                          I'm afraid you cannot proceed further in the experiment as a result.
                          Please return your submission on Prolific now by selecting the "Stop without completing" button, to avoid receiving a rejection.`);

                      } else {
                        $('#jspsych-target').html(`<h1>End of study</h1>\n<p>\nThank you for your interest in this study.
                          However, it is crucial that you understand these instructions to take part, and our data indicate that this is not the case.
                          I'm afraid you cannot proceed further in the experiment as a result.
                          To receive the base pay please enter the secret completion code on prolific:  ` + SECRET_COMPLETION_CODE);
                      }
                    }
                    else{
                      console.log('wrong way2');
                      alert(`You got at least one question wrong. We'll send you back to the instructions and then you can try again.`);
                      return true; // try again
                    }
                }
            }
            REPETITIONS = 0;
            psiturk.recordUnstructuredData('exclusion', false);
            psiturk.finishInstructions();
            psiturk.saveData();
            return false;
        }
    };

    let mortgage_intro = {
      type: 'instructions',
      show_clickable_nav: true,
      pages: function () {
        return [
              `<h1> Gameplay phase </h1>
              You will now play 8 rounds of the Mortgage game.
              Remember, the better you perform, the higher your bonus will be. Have Fun!
              </div>`
            ]
          }
      }

    let mortgage_procedural = {
      type: 'instructions',
      show_clickable_nav: true,
      pages: function(){
        return with_instructions ? [markdown(`<h1>Advice for scoring a high bonus</h1> To help you score higher in the mortgage game, we will show you its near-optimal strategy. This strategy describes in what order to explore the interest rates. Please take a moment to understand this advice and how you could apply it in the game: <br><p style='color: #c73022'>Click the most long-term interest rates that you have not clicked yet. <br> Repeat this step until all the long-term interest rates are clicked or you have encountered the lowest possible interest rate. </p>`)] : [markdown("<h1>Instructions</h1> Good luck in finding a cheap mortgage!")]
      }
    }

    let mortgage_trials = {
        type: 'mortgage',
        blockName: 'test',
        timeline: DEBUG ? getMortgage(2) : getMortgage(8),
        stimulus: '<h1>Mortgage game</h1>',
        plan_buttons: ['Mortgage plan A', 'Mortgage plan B', 'Mortgage plan C'],
        interest_rate_buttons: [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        prompt: '<ul id="mortgage-prompt" style="text-align: left; font-size: 0.8em"><li>Please click either on the blue tiles to find out the interest rates or select a mortgage plan by clicking on the grey tiles.</li> <li>You can click up to 3 interest rates after which you have to choose a plan. </li></ul>',
        response_ends_trial: false,
        max_number_of_clicks: 3,
        interest_rate_button_html: '<button class="jspsych-btn mortgage-ratebutton", style="width: 150px; height: 100px;">%interest_rate_buttons%</button>',
        plan_button_html: '<button class="mortgage-planbutton jspsych-btn", style="width: 150px; height: 100px;">%plan_buttons%</button>',
        start_score: 0,
        trialCount: function () {
            return trialCount;
        },
        on_finish: function () {
            return trialCount += 1;
        },
        show_instructions: with_instructions,
        instructions_html: '<div id="instructions" style="text-align: center; font-weight: 400; margin-bottom: -3em;">Advice for scoring a high bonus: <p style="text-align: center; font-weight: 700; color: #c73022"> Click the most long-term interest rates that you have not clicked yet. <br> Repeat this step until all the long-term interest rates are clicked or you have encountered the lowest possible interest rate.</p></div>'
    };

    let mortgage_testblock = {
      timeline: with_instructions ? [mortgage_procedural, mortgage_intro, mortgage_trials] : [mortgage_intro, mortgage_trials]
    }

    // ---------------------------------------------------------------------------------------------
    // ------------------------------------------- FINISH --------------------------------------------
    // ---------------------------------------------------------------------------------------------

    let divider2 = {
      type: 'instructions',
      show_clickable_nav: true,
      pages: function () {
          return [
              `<h1> End of Game 2:</h1>
              <div style="text-align: left">
                  <li>Please, click the buton below to continue.</li>
              </div>`
            ]
      }
    }

    let demographics = {
        type: 'survey-html-form',
        preamble: "<h1>Demographics</h1> <br> Thanks for participating. We hope you had fun! Please answer the following questions.",
        html: `<div style="text-align: left; margin-bottom: 2px"> <p>
    <strong>What is your gender?</strong><br>
    <input required type="radio" name="gender" value="male"> Male<br>
    <input required type="radio" name="gender" value="female"> Female<br>
    <input required type="radio" name="gender" value="other"> Other<br>
    </p>
    <br>
    <p>
    <strong>How old are you?</strong><br>
    <input required type="number" name="age">
    </p>
    <br>
    <p>
    <strong>Have you ever participated in this (Web of Cash) or a smiliar game (Tree of Cash) on Prolific or MTurk before? We approve and pay you regardless of your answer.</strong><br>
    <input required type="radio" name="naive" value="yes"> Yes<br>
    <input required type="radio" name="naive" value="no">  No<br>
    <p>
    <\div>`
  };

    let finish =  {
      type: 'survey-text',
      preamble: function() {
        return markdown(`# You've completed the experiment\n\n Feel free to give us feedback below before you submit the experiment.\n\n You'll be awarded a bonus based on your performance in 24 hours after the end of the experiment. \n\n Thank you for participating! Hope you enjoyed!
        <br> <br>
        The secret code is <strong>` + SECRET_COMPLETION_CODE + ` </strong>. <br>Please press the 'Finish experiment' button once you've copied it down to paste in the original window.`);
      },
      questions: [
        {prompt: 'Any comments/feedback?'}
      ],
      button_label: 'Finish experiment'
    };


    // timeline
    experiment_timeline = function(){

      let experiment_timeline = [welcome];
      let roadtrip_timeline = [roadtrip_instruct_loop, roadtrip_testblock];
      let mortgage_timeline = [mortgage_instruct_loop, mortgage_testblock];
      let finish_timeline = [divider2, demographics, finish];

      if(with_roadtrip_first){
        roadtrip_timeline.push(divider)
        return experiment_timeline.concat(roadtrip_timeline, mortgage_timeline, finish_timeline)
      } else {
        mortgage_timeline.push(divider)
        return experiment_timeline.concat(mortgage_timeline, roadtrip_timeline, finish_timeline)
      }
    }()

    flatten_timeline = function(timeline){
      var global_timeline = [];

      for(var i in timeline){
        t = timeline[i];

        if(t.timeline !== undefined){
          //recursive for sub timelines
          global_timeline.push( flatten_timeline( t.timeline ));
        } else {
          // its a real block
          if(t.type !== undefined){
            info = t.type;

            if(t.questions !== undefined){
              //info = info + ' : ' + t.questions.toString();
            }
            global_timeline.push( info);

          } else if (t.trial_id !== undefined){
            global_timeline.push( 'Mouselab : ' + t.trial_id)
          }
        }
      }
      global_timeline = [global_timeline.flat(1)];
      return(global_timeline);
    }
    psiturk.recordUnstructuredData('global_timeline', JSON.stringify(flatten_timeline(experiment_timeline)) );
    //console.log(JSON.stringify(flatten_timeline(experiment_timeline)));


// ================================================ #
// ========= START AND END THE EXPERIMENT ========= #
// ================================================ #

// bonus is the (roughly) total score multiplied by something, bounded by min and max amount
    calculateBonus = function () {
        var bonus;
        // Need to deduct the start SCORE of 50 from the roadtrip task
        bonus = SCORE * PARAMS.bonusRate + BONUS;
        bonus = (Math.round(bonus * 100)) / 100; // round to nearest cent
        return Math.min(Math.max(0, bonus), MAX_AMOUNT);
    };

//saving, finishing functions
    reprompt = null;
    save_data = function () {
        return psiturk.saveData({
            success: async function () {
                console.log('Data saved to psiturk server.');
                if (reprompt != null) {
                    window.clearInterval(reprompt);
                }
                await completeExperiment(uniqueId); // Encountering an error here? Try to use Coffeescript 2.0 to compile.
                psiturk.completeHIT();
                return psiturk.computeBonus('compute_bonus');
            },
            error: function () {
                return prompt_resubmit;
            }
        });
    };
    prompt_resubmit = function () {
        $('#jspsych-target').html(`<h1>Oops!</h1>
                                  <p>
                                  Something went wrong submitting your HIT.
                                  This might happen if you lose your internet connection.
                                  Press the button to resubmit.
                                  </p>
                                  <button id="resubmit">Resubmit</button>`);
        return $('#resubmit').click(function () {
            $('#jspsych-target').html('Trying to resubmit...');
            reprompt = window.setTimeout(prompt_resubmit, 10000);
            return save_data();
        });
    };
// initialize jspsych experiment -- without this nothing happens
    return jsPsych.init({
        display_element: 'jspsych-target',
        timeline: experiment_timeline,
        // show_progress_bar: true
        on_finish: function () {
          // adjust roadtrip bonus:
          BONUS = BONUS * 0.2;
          psiturk.recordUnstructuredData('final_bonus', calculateBonus());
          psiturk.recordUnstructuredData('roadtrip_bonus', BONUS);
          psiturk.recordUnstructuredData('mortgage_bonus', SCORE * PARAMS.bonusRate);

            if (DEBUG) {
                save_data();
                return jsPsych.data.displayData();
            } else {
                return save_data();
            }
        },
        on_data_update: function (data) {
            // console.log 'data', data
            psiturk.recordTrialData(data);
            return psiturk.saveData();
        }
    });
};
