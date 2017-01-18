
# coding: utf-8

import sys
import os

baseDir         = os.path.join('data')
ratingsFilename = os.path.join(baseDir, 'ratings.dat')
moviesFilename  = os.path.join(baseDir, 'movies.dat')

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies  = sc.textFile(moviesFilename)

def get_ratings_tuple(entry):
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])

def get_movie_tuple(entry):
    items = entry.split('::')
    return int(items[0]), items[1]

ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()


print 'Ratings: %s' % ratingsRDD.take(2)
print 'Movies: %s' % moviesRDD.take(2)


movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda (user_id,movie_id,rating): (movie_id,[rating]))
                          .reduceByKey(lambda a,b: a+b))

def getCountsAndAverages(RatingsTuple):
    total = 0.0
    for rating in RatingsTuple[1]:
        total += rating
    return (RatingsTuple[0],(len(RatingsTuple[1]),total/len(RatingsTuple[1])))

movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)

movieNameWithAvgRatingsRDD = (moviesRDD
                .join(movieIDsWithAvgRatingsRDD)
                .map(lambda (movieid,(name,(ratings, average))): (average, name, ratings)))


print 'movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3)


def sortFunction(tuple):
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)

movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda (average, name, ratings): ratings > 500)
                                    .sortBy(sortFunction, False))


print 'Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20)


trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)

print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                    validationRDD.count(),
                                                    testRDD.count())

import math

def computeError(predictedRDD, actualRDD):
    predictedReformattedRDD = (predictedRDD
            .map(lambda (UserID, MovieID, Rating):((UserID, MovieID), Rating)) )
                               
    actualReformattedRDD = (actualRDD
            .map(lambda (UserID, MovieID, Rating):((UserID, MovieID), Rating)) )
    
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                        .map(lambda (k,(a,b)): math.pow((a-b),2)))

    totalError = squaredErrorsRDD.reduce(lambda a,b: a+b)
    numRatings = squaredErrorsRDD.count()

    return math.sqrt(float(totalError)/numRatings)


from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda (UserID, MovieID, Rating): (UserID, MovieID))

ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=5L, iterations=5, lambda_=0.1)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank


myModel = ALS.train(trainingRDD, 8, seed=5L, iterations=5, lambda_=0.1)

testForPredictingRDD = testRDD.map(lambda (UserID, MovieID, Rating): (UserID, MovieID))

predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE


myRatedMovies = [
    (0, 845,5.0),  
    (0, 789,4.5),  
    (0, 983,4.8),  
    (0, 551,2.0),  
    (0,1039,2.0),  
    (0, 651,5.0),  
    (0,1195,4.0),  
    (0,1110,5.0),  
    (0,1250,4.5),  
    (0,1083,4.0)
    ]
myRatingsRDD = sc.parallelize(myRatedMovies)


trainingWithMyRatingsRDD = myRatingsRDD.union(trainingRDD)
myRatingsModel = ALS.train(trainingWithMyRatingsRDD, 8, seed=5L, iterations=5, lambda_=0.1)
predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
testRMSEMyRatings = computeError(testRDD, predictedTestMyRatingsRDD)

print 'The model had a RMSE on the test set of %s' % testRMSEMyRatings


myUnratedMoviesRDD = (moviesRDD
                      .map(lambda (movieID, name): movieID)
                      .filter(lambda movieID: movieID not in [ mine[1] for mine in myRatedMovies] )
                      .map(lambda movieID: (0, movieID)))

predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)
print predictedRatingsRDD.take(1)


movieCountsRDD = (movieIDsWithAvgRatingsRDD
                  .map(lambda (MovieID, (ratings, average)): (MovieID, ratings)) )

predictedRDD = predictedRatingsRDD.map(lambda (uid, movie_id, rating): (movie_id, rating))

predictedWithCountsRDD = (predictedRDD.join(movieCountsRDD))

ratingsWithNamesRDD = (predictedWithCountsRDD
                       .join(moviesRDD)
                       .map(lambda (movieID, ((pred, ratings), name)): (pred, name, ratings) )
                       .filter(lambda (pred, name, ratings): ratings > 75))

predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[0])

print ('My highest rated movies as predicted (for movies with more than 75 reviews):\n%s' %
        '\n'.join(map(str, predictedHighestRatedMovies)))

