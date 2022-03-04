const example_graph = {
  start: {A: 5, B: 2},
  A: {C: 4, D: 2},
  B: {A: 8, D: 7},
  C: {D: 6, finish: 3},
  D: {finish: 1},
  finish: {}
};

const findLowestCostNode = (costs, processed) => {
  const knownNodes = Object.keys(costs)
  
  const lowestCostNode = knownNodes.reduce((lowest, node) => {
      if (lowest === null && !processed.includes(node)) {
        lowest = node;
      }
      if (costs[node] < costs[lowest] && !processed.includes(node)) {
        lowest = node;
      }
      return lowest;
  }, null);

  return lowestCostNode
};

// function that returns the minimum cost and path to reach Finish
const dijkstra = (graph) => {
  console.log('Graph: ')
  console.log(graph)

  // track lowest cost to reach each node
  const trackedCosts = Object.assign({finish: Infinity}, graph.start);
  console.log('Initial `costs`: ')
  console.log(trackedCosts)

  // track paths
  const trackedParents = {finish: null};
  for (let child in graph.start) {
    trackedParents[child] = 'start';
  }
  console.log('Initial `parents`: ')
  console.log(trackedParents)

  // track nodes that have already been processed
  const processedNodes = [];

  // Set initial node. Pick lowest cost node.
  let node = findLowestCostNode(trackedCosts, processedNodes);
  console.log('Initial `node`: ', node)

  console.log('while loop starts: ')
  while (node) {
    console.log(`***** 'currentNode': ${node} *****`)
    let costToReachNode = trackedCosts[node];
    let childrenOfNode = graph[node];
  
    for (let child in childrenOfNode) {
      let costFromNodetoChild = childrenOfNode[child]
      let costToChild = costToReachNode + costFromNodetoChild;
  
      if (!trackedCosts[child] || trackedCosts[child] > costToChild) {
        trackedCosts[child] = costToChild;
        trackedParents[child] = node;
      }

      console.log('`trackedCosts`', trackedCosts)
      console.log('`trackedParents`', trackedParents)
      console.log('----------------')
    }
  
    processedNodes.push(node);

    node = findLowestCostNode(trackedCosts, processedNodes);
  }
  console.log('while loop ends: ')

  let optimalPath = ['finish'];
  let parent = trackedParents.finish;
  while (parent) {
    optimalPath.push(parent);
    parent = trackedParents[parent];
  }
  optimalPath.reverse();

  const results = {
    distance: trackedCosts.finish,
    path: optimalPath
  };

  return results;
};