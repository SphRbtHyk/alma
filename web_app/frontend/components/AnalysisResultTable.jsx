import './AlignmentDisplayBlock.css'


const AnalysisResultTab = ({ results, refTextArray, compTextArray }) => {
   let rowCount = 1
   const typeRefDict = new Map()
   typeRefDict.set(0, "MISMATCH")
   typeRefDict.set(1, "ADDITION OR DELETION")
   typeRefDict.set(2, "ADDITION OR DELETION (transposition)")
   typeRefDict.set(3, "TRANSPOSITION")
   let sortedUnmatchesRef = []
   let sortedUnmatchesComp = []
   results.unmatches.map((e, _) => {
     if(Number.isInteger(e[0])){
       sortedUnmatchesRef.push(Number.parseInt(e[0]))
     }
     else{
       sortedUnmatchesComp.push(Number.parseInt(e[1]))
     }
   })
   sortedUnmatchesRef.sort((a, b) => (a > b ? 0 : -1))
   sortedUnmatchesComp.sort((a, b) => (a > b ? 0 : -1))
   let sortedMatches = results.matches.sort((a, b) => (a[0] > b[0] ? 0 : -1))
   let transpositions = new Array()
   for(let i=0;i<sortedMatches.length;i++){
    if(Number.parseInt(sortedMatches[i][0]) !== Number.parseInt(sortedMatches[i][1])){
      transpositions.push(sortedMatches[i])
    }
   }
   
    const genTableBody = (results, refTextArray, compTextArray) => {
       // Order unmatches
       let sortedUnmatchesRef = []
       let sortedUnmatchesComp = []
       results.unmatches.map((e, _) => {
         if(Number.isInteger(e[0])){
           sortedUnmatchesRef.push(parseInt(e[0]))
         }
         else{
           sortedUnmatchesComp.push(parseInt(e[1]))
         }
       })
       sortedUnmatchesRef.sort((a, b) => (a > b ? 0 : -1))
       sortedUnmatchesComp.sort((a, b) => (a > b ? 0 : -1))
       let sortedMatches = results.matches.sort((a, b) => (a[0] > b[0] ? 0 : -1))

       return(
        <tbody>
          {sortedUnmatchesRef.map((element, _) => {
            let id = rowCount
            rowCount++
            return(
              <tr>
                <td className="idCol">{id}</td>
                <td className="typeCol">ADDITION OR DELETION</td>
                <td className="refIdxCol">{element}</td>
                <td className="compInxCol">{element}</td>
                <td className="refTokenCol">-</td>
                <td className="compTokenCol">{refTextArray[element]}</td>
              </tr>
            )
          })}
          {sortedUnmatchesComp.map((element, _) => {
            let id = rowCount
            rowCount++
            return(
              <tr>
                <td className="idCol">{id}</td>
                <td className="typeCol">ADDITION OR DELETION</td>
                <td className="refIdxCol">{element}</td>
                <td className="compInxCol">{element}</td>
                <td className="refTokenCol">{compTextArray[element]}</td>
                <td className="compTokenCol">-</td>
              </tr>
            )
          })}
          {transpositions.map((element, _) => {
            let id = rowCount
            rowCount++
            return(
              <tr>
                <td className="idCol">{id}</td>
                <td className="typeCol">TRANSPOSITION</td>
                <td className="refIdxCol">{element[0]}</td>
                <td className="compInxCol">{element[1]}</td>
                <td className="refTokenCol">{compTextArray[element[0]]}</td>
                <td className="compTokenCol">{compTextArray[element[1]]}</td>
              </tr>
            )
          })}
        </tbody>
       )
    }

    return(
        <div id="summaryTableDiv">
          <h2>Summary table</h2>
          <div>
            <table id="summaryTable">
              <thead>
                <tr>
                  <td className="idCol">ID</td>
                  <td className="typeCol">Type</td>
                  <td className="refIdxCol">Reference index</td>
                  <td className="compInxCol">Compared index</td>
                  <td className="refTokenCol">Reference token</td>
                  <td className="compTokenCol">Compared token</td>
                </tr>
              </thead>
                {genTableBody(results, refTextArray, compTextArray)}
            </table>
          </div>
        </div>
    )
}

export default AnalysisResultTab;
